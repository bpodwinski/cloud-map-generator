# Changelog

## [0.9.0](https://github.com/bpodwinski/cloud-map-generator/compare/v0.8.0...v0.9.0) (2026-11-12)


### Features

* replace match_to_earth with refine_to_earth for improved statistical calibration ([d9e4b96](https://github.com/bpodwinski/cloud-map-generator/commit/d9e4b96ebf0e53e0d2bb657d5451fb1f7b9de37e))

## [0.8.0](https://github.com/bpodwinski/cloud-map-generator/compare/v0.7.0...v0.8.0) (2026-11-12)


### Features

* implement cancellation for ongoing generation with API endpoint and UI support ([ca1bc2d](https://github.com/bpodwinski/cloud-map-generator/commit/ca1bc2dbf9a80fcd3d0cb671cf3668012a3271b8))

## [0.7.0](https://github.com/bpodwinski/cloud-map-generator/compare/v0.6.0...v0.7.0) (2026-11-12)


### Features

* enhance realism tuning with polar filtering and backtrace adjustments ([81aad4a](https://github.com/bpodwinski/cloud-map-generator/commit/81aad4a23b50ce2ca0ea529086c30c52a66c3d7f))

## [0.6.0](https://github.com/bpodwinski/cloud-map-generator/compare/v0.5.0...v0.6.0) (2026-11-12)


### Features

* enhance banded map processing with direct sink communication and improved memory management ([66e327a](https://github.com/bpodwinski/cloud-map-generator/commit/66e327a6b050f12369cb6ac7e697201db3669801))

## [0.5.0](https://github.com/bpodwinski/cloud-map-generator/compare/v0.4.0...v0.5.0) (2026-11-12)


### Features

* implement banded engine for large maps in WASM ([b24a26e](https://github.com/bpodwinski/cloud-map-generator/commit/b24a26ee2bff6270b4f779c2c90d5b55ac224058))
* implement disk-based storage for banded map files in WebAssembly ([e160b01](https://github.com/bpodwinski/cloud-map-generator/commit/e160b010c7dae7bce86951b56f1518c2e90d946d))

## [0.4.0](https://github.com/bpodwinski/cloud-map-generator/compare/v0.3.0...v0.4.0) (2026-11-12)


### Features

* add build stamp to URLs for caching consistency in WebAssembly and worker scripts ([bbdb945](https://github.com/bpodwinski/cloud-map-generator/commit/bbdb94586d107c2bad9319ea002f1724ab557209))

## [0.3.0](https://github.com/bpodwinski/cloud-map-generator/compare/v0.2.0...v0.3.0) (2026-11-12)


### Features

* add OpenAPI documentation and Swagger UI for Cloud Map Generator API ([cf9b8f5](https://github.com/bpodwinski/cloud-map-generator/commit/cf9b8f53bf54ff5fa1d6b17e038b0cbec5073d48))

## [0.2.0](https://github.com/bpodwinski/cloud-map-generator/compare/v0.1.0...v0.2.0) (2026-11-12)


### Features

* add GitHub Actions workflows for site build and deployment ([3feff97](https://github.com/bpodwinski/cloud-map-generator/commit/3feff976cfd0812b639d6a9138263911a7df101c))
* add ground color presets for different celestial bodies in globe rendering ([f191a2a](https://github.com/bpodwinski/cloud-map-generator/commit/f191a2ad5318e62c615d7d5cfc94d8f405f88cab))
* add release-please configuration and manifest for automated versioning ([675c8c1](https://github.com/bpodwinski/cloud-map-generator/commit/675c8c154df94b0ec7a20ca933b68d279e5a9494))
* implement WebGPU support for globe rendering and optimize WebAssembly build process ([9ca75b8](https://github.com/bpodwinski/cloud-map-generator/commit/9ca75b853fdb738434f5b4b491949a3b7331f1a1))


### Bug Fixes

* update wasm file name to match new naming convention ([cde2fd8](https://github.com/bpodwinski/cloud-map-generator/commit/cde2fd8959828b8647f325ac60a19718f460d0e6))
