


cargo build --release --target wasm32-unknown-unknown && wasm-bindgen --out-name luna --out-dir target/luna-web --target web target/wasm32-unknown-unknown/release/luna.wasm  && cp index.html target/luna-web/ && cp -r assets target/luna-web/ && basic-http-server target/luna-web/