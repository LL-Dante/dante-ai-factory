"""All transports are deterministic fixtures; no runtime or Internet is needed."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import httpx
from pydantic import ValidationError
from dante.contracts import ModelRef, LifecycleState, CostClass, PrivacyClass, Task, ToolManifest, RouteDecision
from dante.contracts.runtime import LocalModelMetadata, RuntimeProfile
from dante.contracts.inference import InferenceRequest, AssistantMessage, ToolCall, ToolResult, ToolDefinition
from dante.config import DanteConfig
from dante.local_runtime import OllamaAdapter, LlamaCppAdapter, configured_adapters
from dante.inference import (InferenceGateway, InferenceTimeout, AdapterUnavailable, InvalidResponse,
                             ContextExceeded, PolicyDenied, DeterministicFreeAdapter)
from dante.registry import ModelRegistry
from dante.routing import RuleBasedRouter, RouteDenied
from dante.privacy import PrivacyGate
from dante.agent_host import AgentHostFoundation
from dante.ledger import TaskLedger
from dante.tool_broker import ToolBroker
from dante.task_queue import TaskQueue, ExecutionPlan, Action
from dante.worker import Worker


PIN = 'a' * 64


def model(runtime='ollama', **changes):
    metadata = LocalModelMetadata(exact_identity='fixture-revision-1', runtime_reference='fixture:1', runtime_digest=PIN,
        format='gguf', quantization='fixture', context_tokens=4096, tool_use=True,
        license_id='fixture-license', license_status='verified', machine_profiles={'old-pc'},
        qualification='qualified', identity_verified=True, identity_limitations=('runtime reports identity; file not exposed',),
        eval_refs=('fixture-eval',))
    values = dict(model_id='local-fixture', provider_id=runtime, logical_alias='logical-only', version='1',
        runtime=runtime, local=True, cost_class=CostClass.LOCAL_COMPUTE, lifecycle=LifecycleState.APPROVED,
        capabilities={'tool_calling'}, privacy_eligibility=set(PrivacyClass), local_metadata=metadata)
    values.update(changes)
    return ModelRef(**values)


def request(runtime='ollama', **changes):
    values = dict(model=model(runtime), messages=[{'role':'user','content':'fixture'}])
    values.update(changes)
    return InferenceRequest(**values)


def ollama(content='hello', calls=None):
    return {'model':'fixture:1', 'done':True, 'done_reason':'stop',
            'message':{'role':'assistant','content':content, 'tool_calls':calls or []}}


def llama(content='hello', calls=None):
    return {'model':'fixture:1', 'choices':[{'message':{'role':'assistant','content':content,'tool_calls':calls or []},
                                          'finish_reason':'tool_calls' if calls else 'stop'}]}


def call(name='read', args=None):
    return {'function':{'name':name,'arguments':{} if args is None else args}}


def lcall(args='{}'):
    return {'id':'call-1','type':'function','function':{'name':'read','arguments':args}}


class Wire:
    def __init__(self, runtime='ollama', payload=None):
        self.runtime=runtime
        self.payload = payload if payload is not None else (ollama() if runtime=='ollama' else llama())
        self.calls=[]
        self.error=None
        self.status=200
        self.pin=PIN
        self.models=True
        self.remote=False

    def __call__(self, req):
        self.calls.append((req.url.path, json.loads(req.content) if req.content else None))
        if self.error:
            raise self.error
        if req.url.path=='/api/tags':
            data={'models':[{'name':'fixture:1','digest':self.pin,'details':{'format':'gguf'}}] if self.models else []}
        elif req.url.path=='/api/show':
            data={'details':{'format':'gguf'},'model_info':{'general.architecture':'fixture'}}
            if self.remote: data['remote_host']='remote.example'
        elif req.url.path=='/health': data={'status':'ok'}
        elif req.url.path=='/v1/models': data={'data':[{'id':'fixture:1'}] if self.models else []}
        else:
            return httpx.Response(self.status, json=self.payload)
        return httpx.Response(200,json=data)

    def adapter(self):
        cls=OllamaAdapter if self.runtime=='ollama' else LlamaCppAdapter
        return cls(machine_profile='old-pc', transport=httpx.MockTransport(self))


class LocalAdapterTests(unittest.TestCase):
    def test_ollama_text_unknown_cost_and_usage(self):
        result=Wire().adapter().complete(request())
        self.assertEqual(result.content,'hello')
        self.assertIsNone(result.cost)
        self.assertIsNone(result.usage.input_tokens)
        self.assertEqual(result.finish_reason,'stop')

    def test_ollama_usage_preserved_not_summed(self):
        payload=ollama()
        payload.update(prompt_eval_count=10, eval_count=3)
        result=Wire(payload=payload).adapter().complete(request())
        self.assertEqual(result.usage.input_tokens,10)
        self.assertEqual(result.usage.output_tokens,3)
        self.assertIsNone(result.usage.total_tokens)

    def test_ollama_one_tool_call(self):
        result=Wire(payload=ollama('',[call(args={'nested':{'n':2}})])).adapter().complete(request())
        self.assertEqual(result.tool_calls[0].arguments,{'nested':{'n':2}})
        self.assertTrue(result.tool_calls[0].call_id)
        self.assertEqual(result.finish_reason,'tool_calls')

    def test_ollama_two_tool_calls_unique_ids(self):
        result=Wire(payload=ollama('',[call(),call('write')])).adapter().complete(request())
        self.assertEqual(len({c.call_id for c in result.tool_calls}),2)

    def test_ollama_malformed_args(self):
        for args in ('{}', [], 1):
            with self.subTest(args=args), self.assertRaises(InvalidResponse):
                Wire(payload=ollama('',[call(args=args)])).adapter().complete(request())

    def test_ollama_malformed_or_empty_response(self):
        for payload in ({}, ollama(''), {'done':False}, ollama(calls=[{}])):
            with self.subTest(payload=payload), self.assertRaises(InvalidResponse):
                Wire(payload=payload).adapter().complete(request())

    def test_ollama_timeout(self):
        wire=Wire(); wire.error=httpx.ReadTimeout('fixture')
        with self.assertRaises(InferenceTimeout): wire.adapter().complete(request())

    def test_ollama_unavailable(self):
        wire=Wire(); wire.error=httpx.ConnectError('fixture')
        with self.assertRaises(AdapterUnavailable): wire.adapter().complete(request())

    def test_ollama_context_error(self):
        wire=Wire(payload={'error':'input exceeds the context window'}); wire.status=400
        with self.assertRaises(ContextExceeded): wire.adapter().complete(request())

    def test_ollama_continuation_and_no_internal_metadata(self):
        wire=Wire()
        assistant=AssistantMessage(tool_calls=(ToolCall(call_id='stable',name='read',arguments={'path':'a'}),))
        req=request(messages=[{'role':'user','content':'x'},assistant,ToolResult(call_id='stable',content='done')],
                    tools=(ToolDefinition(name='read',parameters={'type':'object'}),), task_id='internal-task',
                    trace_id='internal-trace',max_output_tokens=4)
        wire.adapter().complete(req)
        body=wire.calls[-1][1]
        self.assertEqual(body['messages'][-1]['tool_name'],'read')
        self.assertEqual(body['model'],'fixture:1')
        self.assertEqual(body['options']['num_predict'],4)
        self.assertNotIn('internal-',json.dumps(body))
        self.assertNotIn('local_metadata',json.dumps(body))

    def test_ollama_runtime_digest_change_and_cloud_denied(self):
        wire=Wire(); wire.pin='b'*64
        with self.assertRaises(PolicyDenied): wire.adapter().complete(request())
        self.assertNotIn('/api/chat',[p for p,_ in wire.calls])
        wire=Wire(); wire.remote=True
        with self.assertRaises(PolicyDenied): wire.adapter().complete(request())
        self.assertNotIn('/api/chat',[p for p,_ in wire.calls])

    def test_llamacpp_text_and_usage(self):
        payload=llama(); payload['usage']={'prompt_tokens':3,'completion_tokens':4,'total_tokens':7}
        result=Wire('llamacpp',payload).adapter().complete(request('llamacpp'))
        self.assertEqual(result.content,'hello'); self.assertEqual(result.usage.total_tokens,7)
        self.assertIsNone(result.cost)

    def test_llamacpp_tool_call(self):
        result=Wire('llamacpp',llama('',[lcall('{"x":2}')])).adapter().complete(request('llamacpp'))
        self.assertEqual(result.tool_calls[0].arguments,{'x':2})
        self.assertEqual(result.tool_calls[0].call_id,'call-1')

    def test_llamacpp_malformed_response(self):
        for payload in ({},llama(''),llama('',[lcall('[]')]),llama('',[lcall('{"x":1,"x":2}')])):
            with self.subTest(payload=payload),self.assertRaises(InvalidResponse):
                Wire('llamacpp',payload).adapter().complete(request('llamacpp'))

    def test_llamacpp_timeout(self):
        wire=Wire('llamacpp'); wire.error=httpx.ReadTimeout('fixture')
        with self.assertRaises(InferenceTimeout): wire.adapter().complete(request('llamacpp'))

    def test_llamacpp_unavailable(self):
        wire=Wire('llamacpp'); wire.error=httpx.ConnectError('fixture')
        with self.assertRaises(AdapterUnavailable): wire.adapter().complete(request('llamacpp'))

    def test_llamacpp_context_error(self):
        wire=Wire('llamacpp',{'error':{'code':'context_length_exceeded'}}); wire.status=400
        with self.assertRaises(ContextExceeded): wire.adapter().complete(request('llamacpp'))

    def test_health_reachable_present_missing_and_unavailable(self):
        for runtime in ('ollama','llamacpp'):
            wire=Wire(runtime); adapter=wire.adapter()
            health=adapter.health(model(runtime))
            self.assertTrue(health.runtime_reachable and health.model_present and health.adapter_operational)
            self.assertIsNotNone(health.observed_at)
            wire.models=False
            self.assertFalse(adapter.health(model(runtime)).model_present)
            wire.error=httpx.ConnectError('fixture')
            self.assertFalse(adapter.health().runtime_reachable)

    def test_health_unknown_model_identity_stays_unknown(self):
        health=Wire().adapter().health(model(local_metadata=LocalModelMetadata()))
        self.assertIsNone(health.model_present)
        self.assertEqual(health.error,'model_reference_unknown')

    def test_endpoint_policy_and_redirect_denied(self):
        for url in ('http://example.com','http://127.0.0.1.evil','http://u:p@localhost','http://127.0.0.1/path'):
            with self.assertRaises(ValidationError): RuntimeProfile(runtime='ollama',base_url=url)
        self.assertEqual(RuntimeProfile(runtime='ollama',base_url='http://localhost:11434').base_url,'http://127.0.0.1:11434')
        self.assertTrue(RuntimeProfile(runtime='ollama',base_url='http://192.0.2.1',allow_remote=True).allow_remote)
        wire=Wire(); wire.status=302
        with self.assertRaises(PolicyDenied): wire.adapter().complete(request())

    def test_direct_adapter_unqualified_model_denied_before_http(self):
        wire=Wire(); candidate=model(lifecycle=LifecycleState.CANDIDATE)
        with self.assertRaises(PolicyDenied): wire.adapter().complete(request(model=candidate))
        self.assertEqual(wire.calls,[])

    def test_real_loopback_http_boundary(self):
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        import threading
        wire=Wire('llamacpp')
        class Handler(BaseHTTPRequestHandler):
            def do_GET(self): self.respond()
            def do_POST(self): self.respond()
            def respond(self):
                content=self.rfile.read(int(self.headers.get('Content-Length',0)))
                response=wire(httpx.Request(self.command,'http://127.0.0.1'+self.path,content=content))
                data=response.content
                self.send_response(response.status_code)
                self.send_header('Content-Type','application/json')
                self.send_header('Content-Length',str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            def log_message(self,*args): pass
        server=ThreadingHTTPServer(('127.0.0.1',0),Handler)
        thread=threading.Thread(target=server.serve_forever,daemon=True); thread.start()
        try:
            profile=RuntimeProfile(runtime='llamacpp',base_url=f'http://127.0.0.1:{server.server_port}')
            result=LlamaCppAdapter(profile,machine_profile='old-pc').complete(request('llamacpp'))
            self.assertEqual(result.content,'hello')
        finally:
            server.shutdown(); server.server_close(); thread.join()

    def test_usage_and_supplied_call_ids_are_strict(self):
        payload=ollama('',[dict(call(),id='')])
        with self.assertRaises(InvalidResponse): Wire(payload=payload).adapter().complete(request())
        payload=llama(); payload['usage']=0
        with self.assertRaises(InvalidResponse): Wire('llamacpp',payload).adapter().complete(request('llamacpp'))

    def test_missing_or_changed_file_denied_at_adapter_boundary(self):
        wire=Wire(); m=model()
        with tempfile.TemporaryDirectory() as directory:
            m.local_metadata.local_path=Path(directory)/'missing.gguf'
            m.local_metadata.sha256=PIN
            with self.assertRaises(PolicyDenied): wire.adapter().complete(request(model=m))
        self.assertEqual(wire.calls,[])

    def test_http_unavailable_is_reachable_but_not_operational(self):
        adapter=LlamaCppAdapter(transport=httpx.MockTransport(lambda req:httpx.Response(503,json={'error':'loading'})))
        result=adapter.health()
        self.assertTrue(result.runtime_reachable)
        self.assertFalse(result.adapter_operational)
        self.assertEqual(result.error,'AdapterUnavailable')



class RegistryTests(unittest.TestCase):
    def test_manifest_and_qualified_model_accepted(self):
        m=model(); registry=ModelRegistry([m],machine_profile='old-pc')
        registry.require_automatic(m,tool_use=True)
        self.assertEqual(registry.get(m.model_id).local_metadata.format,'gguf')

    def test_unknown_unqualified_and_missing_license_denied(self):
        for metadata in (None,LocalModelMetadata(),model().local_metadata.model_copy(update={'license_id':None})):
            m=model(local_metadata=metadata)
            with self.assertRaises(ValueError): ModelRegistry(machine_profile='old-pc').require_automatic(m)
        unknown=LocalModelMetadata()
        self.assertIsNone(unknown.license_id); self.assertIsNone(unknown.exact_identity)
        self.assertEqual(unknown.qualification,'unverified')

    def test_digest_mismatch_and_changed_file_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fixture.gguf'; path.write_bytes(b'fixture weights')
            m=model(); m.local_metadata.local_path=path
            m.local_metadata.sha256=hashlib.sha256(path.read_bytes()).hexdigest()
            registry=ModelRegistry([m],machine_profile='old-pc')
            registry.require_automatic(m)
            path.write_bytes(b'changed')
            with self.assertRaises(ValueError): registry.require_automatic(m)
            with self.assertRaises(RouteDenied): RuleBasedRouter(registry).decide(Task(goal='x',workspace=directory),PrivacyGate().classify('x'),set())

    def test_machine_profile_incompatibility(self):
        with self.assertRaises(ValueError): ModelRegistry(machine_profile='future-pc').require_automatic(model())

    def test_lifecycle_promotions_require_evidence(self):
        m=model(lifecycle=LifecycleState.CANDIDATE)
        registry=ModelRegistry([m],machine_profile='old-pc')
        with self.assertRaises(ValueError): registry.promote(m.model_id,LifecycleState.PRODUCTION)
        for stage in (LifecycleState.IMPORTED,LifecycleState.VERIFIED,LifecycleState.EVALUATED,LifecycleState.APPROVED,LifecycleState.PRODUCTION):
            self.assertEqual(registry.promote(m.model_id,stage).lifecycle,stage)

    def test_separate_runtime_machine_configuration(self):
        config=DanteConfig(workspace=Path('workspace'), machine_profile='old-pc',local_runtimes={
            'cpu':RuntimeProfile(runtime='ollama',base_url='http://127.0.0.1:11434')})
        adapter=configured_adapters(config)[0]
        self.assertEqual(adapter.provider_id,'cpu'); self.assertEqual(adapter.machine_profile,'old-pc')

    def test_no_cloud_fallback_after_local_failure(self):
        local=model(); cloud=model(model_id='cloud',provider_id='cloud',local=False,cost_class=CostClass.ZERO)
        registry=ModelRegistry([local,cloud],machine_profile='old-pc')
        decision=RuleBasedRouter(registry).decide(Task(goal='x',workspace='x'),PrivacyGate().classify('x'),set())
        self.assertEqual(decision.candidates,(local.model_id,))
        # Also reject a caller-supplied mixed fallback list at the gateway.
        decision=decision.model_copy(update={'candidates':(local.model_id,cloud.model_id)})
        wire=Wire(); wire.error=httpx.ConnectError('fixture')
        gateway=InferenceGateway(registry,[wire.adapter(),DeterministicFreeAdapter('cloud')])
        with self.assertRaises(AdapterUnavailable): gateway.infer(decision,request())

    def test_worker_generic_local_inference_e2e(self):
        for runtime in ('ollama','llamacpp'):
            with self.subTest(runtime=runtime),tempfile.TemporaryDirectory() as directory:
                root=Path(directory); ledger=TaskLedger(root/'task.db'); queue=TaskQueue(ledger)
                task=queue.submit(Task(goal='local inference',workspace=directory),ExecutionPlan(actions=[Action(tool_id='save')]))
                registry=ModelRegistry([model(runtime)],machine_profile='old-pc')
                gateway=InferenceGateway(registry,[Wire(runtime).adapter()])
                def factory(db,current):
                    broker=ToolBroker(ledger=db)
                    host=AgentHostFoundation(db,PrivacyGate(),RuleBasedRouter(registry),gateway,broker)
                    response=host.infer(current.task_id,'fixture',set())
                    def save():
                        path=root/'result.txt'; path.write_text(response.content)
                        return {'ok':True,'path':str(path)}
                    broker.register(ToolManifest(tool_id='save',permissions={'write_workspace'},risk='R1',
                        filesystem_scope=directory,arguments_schema={'type':'object'}),save,required_permissions={'write_workspace'})
                    return host
                Worker(ledger,factory).run_once()
                self.assertEqual(queue.status(task.task_id)['task_status'],'completed')
                self.assertEqual((root/'result.txt').read_text(),'hello')
                self.assertEqual(len(ledger.steps(task.task_id)),1)


if __name__=='__main__': unittest.main()
