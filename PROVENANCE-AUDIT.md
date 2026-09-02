# File-by-file provenance recheck

Owner-confirmed decision: Apache-2.0. Status: PASS based on the declaration in
LICENSE-DECISION.md and the traceability checks. This does not assert an independent
chain-of-title certification or a stronger warranty than the owner's knowledge.

All listed files were reviewed. Baseline-derived source hashes were verified;
two EOF-only changes have identical ASTs. pyproject.toml subsequently received
license metadata only. No cptr assets or copied patch fragments are included.
The table records the earlier review before documentation/license updates;
SOURCE_MANIFEST.json records current baseline-derived hashes. New LICENSE is the
canonical Apache text. All current files are scanned before the initial commit.

| File | Traceable origin | SHA-256 at recheck |
|---|---|---|
| .env.example | Sanitized export preparation; generated documentation/configuration/manifest | `668fe4f9418690cfa156ab25fdf3d6051f3ffb3bfada9580eb64d7b25efa8d71` |
| .gitignore | Sanitized export preparation; generated documentation/configuration/manifest | `0d0246b11093e5e3046dc265e4af0d1cf05f6643e5a0e3958b113bc38a6a2110` |
| config/dante/base.yaml | Sanitized export preparation; generated documentation/configuration/manifest | `57616f7ed78b17f3aa9175cdd70c94d282b66d36126f2871253e72ef871c7ecf` |
| CONTRIBUTING.md | Sanitized export preparation; generated documentation/configuration/manifest | `d974a579be0834ff9bb402bc3d6049b9c233e4bbb3ce87bf0883db300e626fe3` |
| dante/__init__.py | Private baseline; first tracked 1b2a65a | `4bceec5cd0cc9338d99c529cefa8de065b2db5983a18d680e9e09b7d2743eff3` |
| dante/__main__.py | Private baseline; first tracked a903dc9 | `d423419fcad01e56e20ead95b8ec334776262e9438e15df3912ba36be0411ce4` |
| dante/acceptance.py | Private baseline; first tracked 1b2a65a | `af1bb0fafca84cc0034a26100a45d1db848ac9886377610c2923c3f0d1acc326` |
| dante/agent_host.py | Private baseline; first tracked 1b2a65a | `d961ced558225483726a7aca157663c262a1712b64068c280cc331d3648883e8` |
| dante/approvals.py | Private baseline; first tracked 1b2a65a | `4037445ce3e85e6ad02ddee6a2a13952f469b2492e4778d2cd8e57a4b9cf032a` |
| dante/artifacts.py | Private baseline; first tracked 1b2a65a | `9322d11727b33df95cf8a5d6ff1d4fcbec63d02948823c51930001227a944fa6` |
| dante/cli.py | Private baseline; first tracked a903dc9 | `a7e5bbf1d4245bc029480f6a45523230486871da46c773d6775db4ed1cd28eaa` |
| dante/config.py | Private baseline; first tracked 1b2a65a | `f823426b96e7259f0d02cc922f081b77129c4c6a38ed693e3a8a5d1ac43b19cf` |
| dante/continuity.py | Private baseline; first tracked 8ce432b | `46cb3c10a10fae7c350dab30cd5a68754a25be1f31fcbf399403b65687a20efb` |
| dante/contracts/__init__.py | Private baseline; first tracked 1b2a65a | `0bd55f2ac34e33d7cfbf3f453eb8baee460dd681ff3daf52cef8bd1e94491def` |
| dante/contracts/continuity.py | Private baseline; first tracked 8ce432b | `b9d09590cd16e7f9af2dae5e194b948108a2a4262cd8b0bbb32eaf11216dbfeb` |
| dante/contracts/inference.py | Private baseline; first tracked 4b9362f | `6ba3cbf2164398072d87be536a6f34fcb9a5898cedc7316b117d2d3d3927da1b` |
| dante/contracts/runtime.py | Private baseline; first tracked fe8a60d | `6272d9d4be5c9008674b717b3481b6b616abd593bea856c829db74d52b192163` |
| dante/contracts/tools.py | Private baseline; first tracked 6ec7877 | `867b32baa4f770b71f04d98787a1fee36148c0aaccd9b16f8eb8a56dea1b7c7f` |
| dante/evals.py | Private baseline; first tracked 1b2a65a | `6cbd9d3e037156e29d5180768ad7834857e31241f4cddd93e65ed49a03bb9691` |
| dante/inference.py | Private baseline; first tracked 1b2a65a | `781d618852f9ada047a61066b498ee1c50834ee8f11ad633cd89d0ef64b03d74` |
| dante/ledger.py | Private baseline; first tracked 1b2a65a | `871576681583ef19817f08eb455348de434ddb3256ca704c549edde22b6c7ebf` |
| dante/local_runtime.py | Private baseline; first tracked fe8a60d | `8a55dc7ba97fb84a196090959ada675cb12e4704d7dc2fed69c2bf3a51ead83e` |
| dante/migrations.py | Private baseline; first tracked caf78ca | `f4baa2ba2d58ae903c2902e8fde691cc3c4be3d41ccec2435eb4215ed195c92d` |
| dante/privacy.py | Private baseline; first tracked 1b2a65a | `a2209c79248e7aa36dd203d7bfef0bd7c7a98a9bddffec209a5964a848c84026` |
| dante/recovery.py | Private baseline; first tracked caf78ca | `bf097fd1175c7abb754e3ca77de1839e80ca7fcafc3f4c25ca99d07bcdb38081` |
| dante/registry.py | Private baseline; first tracked 1b2a65a | `6695561593f9ad6e6925b6a4d1ce62c70d14a950f629ab7934a97f860af60e89` |
| dante/routing.py | Private baseline; first tracked 1b2a65a | `ef0fdc742791ae12c7038f9956ef837d03a47b4ad48980484a288013967ea33f` |
| dante/task_queue.py | Private baseline; first tracked a903dc9 | `4a82815437f0eda5985fc68839cceefda47c2c0987b7c76d913889e8d41f7dba` |
| dante/telemetry.py | Private baseline; first tracked 1b2a65a | `0f544ca4e13e605eb07c1d460c7b10b81542d00ea1c9269026e293c3ea4a3ac5` |
| dante/tool_broker.py | Private baseline; first tracked 1b2a65a | `d4ed85a62bf505c49821da64d91a0d8d2767a5ca2bdeb82d5cfd7db56bff91db` |
| dante/worker.py | Private baseline; first tracked a903dc9 | `162d5ef14f6ad209265d34ddd6593ce4221946e3969adae802bdaa1d8e8df75e` |
| EXPORT_QUALIFICATION.md | Sanitized export preparation; generated documentation/configuration/manifest | `ec76cb99e41baac53dc8c57596307fb22be2148f0b406831c111bdd8e4d22e79` |
| LICENSE-DECISION.md | Sanitized export preparation; generated documentation/configuration/manifest | `1f12e6127fba10569022eca9f2632c288b0bee6f71c8d720d25718f9d6a25695` |
| pyproject.toml | Private baseline; first tracked 1b2a65a | `b79a30ef6a81cae8cf29ad512e6be13954d1e34132b198357345ba4b252b3199` |
| README.md | Sanitized export preparation; generated documentation/configuration/manifest | `4abbdf8cdeb670fee114b242a36d46123c6e50fca032a1980c70dcbc946fb704` |
| requirements-dante.in | Private baseline; first tracked 1b2a65a | `bff2b0a7ec320ed529f4820a4f93e220e3cf6b56a559afaaf3ae7595aaac7f0e` |
| requirements-dante.lock | Sanitized export preparation; generated documentation/configuration/manifest | `a8765ecd88fd2fae08924bbdf07095309a937e5d032f584b1d877e5925212671` |
| scripts/dante_e2e.py | Private baseline; first tracked 1b2a65a | `103396673c403aad3403a07f1f47250a80e32e5098a9b24315b270343044d644` |
| SECURITY.md | Sanitized export preparation; generated documentation/configuration/manifest | `78b6d149624c0877d508ce9f0bf240e6c01edfce459e64164a617e5972d48de7` |
| SOURCE_MANIFEST.json | Sanitized export preparation; generated documentation/configuration/manifest | `1b596c689401ef9dd08f38fd343caca00f697c01cd5e98e36f5b973aca46a97c` |
| tests/__init__.py | Private baseline; first tracked 1b2a65a | `f93cfca20e716bdcfa1dc8ca3a6b1afdc990a25826bae58e6576717af4c2e26c` |
| tests/continuity_process.py | Private baseline; first tracked 8ce432b | `496698779b5b606bce838dcde5bcc7a2c33aef100b5d35da5b5aa663a543bc54` |
| tests/recovery_process.py | Private baseline; first tracked caf78ca | `4866c72abb9cd0fb332dea7c7a9dd8cd838e82e2c232dfa890c0f42b6fed3390` |
| tests/test_acceptance.py | Private baseline; first tracked 1b2a65a | `9bbfcce2d0f9505659c4a328dd3e27e0a913aae2fb270c02f9da2d736487e844` |
| tests/test_continuity.py | Private baseline; first tracked 8ce432b | `cc3cede122b288fb59aa98ba5aca7c430a0a0212cbf32109bf06390f9d836a31` |
| tests/test_contracts_config_trace.py | Private baseline; first tracked 1b2a65a | `5930eef515d5c76bff0feae935cc71858f8c3f8c7203255dd4112c0bf63ac66f` |
| tests/test_e2e_resume.py | Private baseline; first tracked 1b2a65a | `0e47fe4fd1d7e5055dcc2a025b5d798eae68bf4f3caac22b9f416c0acde49aaa` |
| tests/test_gateway_tools_artifacts_approval.py | Private baseline; first tracked 1b2a65a | `b5c55d5893f209641c4a9ee619dccf411bd198e5a432f2cd773314204bc8faa7` |
| tests/test_inference_contract.py | Private baseline; first tracked 4b9362f | `4e262ab8106e420bd83631d5abd86d8fec891051f7665f8f59a97822c54c4293` |
| tests/test_ledger_privacy_router.py | Private baseline; first tracked 1b2a65a | `373f61d997a6865acc76fccc7425c6a779387f73e59d730102481a043c45a3ab` |
| tests/test_local_runtime.py | Private baseline; first tracked fe8a60d | `b8932408220fe1db2ddaa13147e07a730a74692d3b523eb7a9179db8d131f64f` |
| tests/test_recovery.py | Private baseline; first tracked caf78ca | `e1edfe80d075ee0a3124f4e55d41216083374e9837e113f5641ffabfabb199e9` |
| tests/test_tool_enforcement.py | Private baseline; first tracked 6ec7877 | `bb739a77ebceb49a2a790f340aa1cbd4dfb5dc107fd781508ca93f9387169720` |
| tests/test_worker.py | Private baseline; first tracked a903dc9 | `9ae87b08a8104bb2ad60c01a2d9b3464076fbc5e82480ec5729ae9ceb1d6c044` |
| tests/worker_process.py | Private baseline; first tracked a903dc9 | `869a50036835c84ceda1671cc60dcfba5b8472a99b08046cf982aea65fbe561d` |
| THIRD_PARTY_NOTICES.md | Sanitized export preparation; generated documentation/configuration/manifest | `a4f24e4db1c01a017947d02c82853a6d04df9b2dbc5e07a4e81b37ff91500e39` |
| tools/ai_cloud_workspace.py | Private baseline; first tracked 504e7ad | `64204a0f67f3213ec4c868b458095a5c2db8d2824c3235a8bd775c11d9fca5db` |

Hashes describe input to this recheck; documentation updated afterward is identified in LICENSE-DECISION.md. This inventory is not a legal ownership certificate.
