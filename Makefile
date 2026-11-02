CPP_SOURCES := $(shell find src benchmark -name '*.cpp' -o -name '*.hpp')
CPP_FLAGS_DIR ?= build/cpp

# The C++ dependencies (a pinned vcpkg checkout and the static Arrow it
# builds) live outside the repository, so a checkout in a synced folder stays
# small. Override FFT_DEPS_DIR to put them elsewhere; CMake reads the same
# variable.
FFT_DEPS_DIR ?= $(HOME)/.cache/fast-feature-transform
VCPKG_ROOT := $(FFT_DEPS_DIR)/vcpkg
VCPKG_INSTALLED_DIR := $(FFT_DEPS_DIR)/vcpkg_installed

# The vcpkg checkout must match builtin-baseline in vcpkg.json: that pair pins
# Arrow, Parquet and everything vcpkg builds for them (such as Thrift) to one
# exact version. CMakeLists.txt pins the header-only libraries and Catch2 by
# hash, and pyproject.toml pins nanobind with the other build requirements.
VCPKG_COMMIT := 61d7eaeb094a6acbc23d9c2e54de9208175fda47
UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)
ifeq ($(UNAME_S),Darwin)
  VCPKG_TRIPLET := $(if $(filter arm64,$(UNAME_M)),arm64-osx,x64-osx)
else
  VCPKG_TRIPLET := $(if $(filter aarch64 arm64,$(UNAME_M)),arm64-linux,x64-linux)
endif

.PHONY: dev python-format python-lint python-test audit parity-fixtures sklearn-fixtures release-requirements transformer-benchmark label-encoder-benchmark cpp-format cpp-lint cpp-tidy cpp-test cpp-sanitize cpp-tsan cpp-flags check vcpkg

dev:
	uv sync

python-format:
	uv run ruff check --select I --fix .
	uv run ruff format .

python-lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run ty check .

python-test:
	uv run pytest ./test

parity-fixtures:
	uv run --isolated --no-project --python 3.14t --with category_encoders==2.10.0 --with pandas \
		python scripts/regen_parity_fixtures.py

sklearn-fixtures:
	uv run --isolated --no-project --python 3.14t --with scikit-learn==1.9.0 --with scipy==1.18.1 --with pandas \
		python scripts/regen_sklearn_fixtures.py

# The hashed lists release.yaml installs from: cibuildwheel and the tools it
# installs for itself, the build requirements, the wheel tests' packages and
# twine, each pinned to one version with its hashes. cibuildwheel.in,
# cibuildwheel-tools.in and twine.in are edited by hand; this copies [tool.uv]
# build-constraint-dependencies to build-constraints.in, and exports the
# wheel-tests dependency group from uv.lock with the lock's own hashes. Run it
# after changing any of those versions or uv.lock;
# test/test_release_requirements.py fails until then. It keeps every other
# version the compiled lists hold: delete those .txt files first to move them
# too. twine's list is for Python 3.12, the twine job's system Python.
release-requirements:  ## the hashed lists release.yaml installs from (needs network access)
	cd .github/requirements && trap 'rm -f build-constraints.in.tmp' EXIT && \
	uv run --no-project python -c "import tomllib; \
	print(*tomllib.load(open('../../pyproject.toml', 'rb'))['tool']['uv']['build-constraint-dependencies'], sep='\n')" \
		> build-constraints.in.tmp && \
	mv build-constraints.in.tmp build-constraints.in && \
	uv export --locked --only-group wheel-tests --no-emit-project --format requirements-txt --quiet \
		--output-file wheel-tests.txt && \
	uv pip compile --quiet --universal --python-version 3.14 --generate-hashes \
		--custom-compile-command "make release-requirements" cibuildwheel.in -o cibuildwheel.txt && \
	uv pip compile --quiet --universal --python-version 3.14 --generate-hashes \
		--custom-compile-command "make release-requirements" cibuildwheel-tools.in -o cibuildwheel-tools.txt && \
	uv pip compile --quiet --universal --python-version 3.14 --generate-hashes \
		--custom-compile-command "make release-requirements" build-constraints.in -o build-constraints.txt && \
	uv pip compile --quiet --universal --python-version 3.12 --generate-hashes \
		--custom-compile-command "make release-requirements" twine.in -o twine.txt

transformer-benchmark:  ## five transformers vs category_encoders and scikit-learn (needs network access)
	uv run --with category_encoders==2.10.0 --with pandas python benchmark/transformers.py

label-encoder-benchmark:  ## LabelEncoder vs scikit-learn
	uv run python benchmark/label_encoder.py

audit:
	uv run bandit -r -c pyproject.toml fast_feature_transform test
	uv audit

cpp-test:
	PATH="$(PWD)/.venv/bin:$$PATH" cmake -S . -B build/cpp -G Ninja -DFFT_BUILD_TESTS=ON -DFFT_ENABLE_SANITIZERS=OFF -DCMAKE_BUILD_TYPE=Debug
	ln -sf build/cpp/compile_commands.json .
	$(MAKE) cpp-flags
	PATH="$(PWD)/.venv/bin:$$PATH" cmake --build build/cpp
	PATH="$(PWD)/.venv/bin:$$PATH" ctest --test-dir build/cpp --output-on-failure

cpp-sanitize:  ## native AddressSanitizer + UndefinedBehaviorSanitizer suite in its own build directory
	PATH="$(PWD)/.venv/bin:$$PATH" cmake -S . -B build/cpp-sanitize -G Ninja -DFFT_BUILD_TESTS=ON -DFFT_ENABLE_SANITIZERS=ON -DCMAKE_BUILD_TYPE=Debug
	$(MAKE) cpp-flags CPP_FLAGS_DIR=build/cpp-sanitize
	PATH="$(PWD)/.venv/bin:$$PATH" cmake --build build/cpp-sanitize
	PATH="$(PWD)/.venv/bin:$$PATH" ctest --test-dir build/cpp-sanitize --output-on-failure

cpp-tsan:  ## native ThreadSanitizer suite in its own build directory (cannot share a build with cpp-sanitize)
	PATH="$(PWD)/.venv/bin:$$PATH" cmake -S . -B build/cpp-tsan -G Ninja -DFFT_BUILD_TESTS=ON -DFFT_ENABLE_THREAD_SANITIZER=ON -DCMAKE_BUILD_TYPE=Debug
	$(MAKE) cpp-flags CPP_FLAGS_DIR=build/cpp-tsan
	PATH="$(PWD)/.venv/bin:$$PATH" cmake --build build/cpp-tsan
	TSAN_OPTIONS="suppressions=$(PWD)/tsan.supp print_suppressions=1" \
		PATH="$(PWD)/.venv/bin:$$PATH" ctest --test-dir build/cpp-tsan --output-on-failure

cpp-flags:  ## every one of our translation units must be compiled with -ffp-contract=off
	@uv run python -c "import json, sys; \
	db = json.load(open('$(CPP_FLAGS_DIR)/compile_commands.json')); \
	ours = [e for e in db if '/src/fastfeaturetransform/' in e['file'] or '/src/bindings/' in e['file']]; \
	missing = [e['file'] for e in ours if '-ffp-contract=off' not in e['command']]; \
	sys.exit('compile_commands.json: %d of %d entries lack -ffp-contract=off:\n  %s' % (len(missing), len(ours), '\n  '.join(missing))) if missing or not ours else print('%d translation units carry -ffp-contract=off' % len(ours))"

cpp-format:
	uv run clang-format -i $(CPP_SOURCES)

cpp-lint:
	uv run clang-format --dry-run -Werror $(CPP_SOURCES)

cpp-tidy:  ## clang-tidy over the C++ test build (run cpp-test first); local-only, the tool is unpinned
	@if command -v run-clang-tidy >/dev/null 2>&1; then \
		run-clang-tidy -p build/cpp -j 8 'src/fastfeaturetransform/'; \
	else \
		echo "clang-tidy not installed (brew install llvm); skipping"; \
	fi

check: python-format python-lint audit python-test cpp-lint cpp-test

vcpkg:  ## one-time C++ dependency setup into $(FFT_DEPS_DIR): pinned vcpkg checkout + private static Arrow
	@if command -v brew >/dev/null 2>&1; then \
		brew install pkg-config bison; \
	else \
		echo "Homebrew not found: make sure pkg-config and bison >= 2.5 are installed"; \
	fi
	mkdir -p "$(FFT_DEPS_DIR)"
	test -d "$(VCPKG_ROOT)/.git" || git init --quiet "$(VCPKG_ROOT)"
	git -C "$(VCPKG_ROOT)" remote get-url origin >/dev/null 2>&1 || git -C "$(VCPKG_ROOT)" remote add origin https://github.com/microsoft/vcpkg
	git -C "$(VCPKG_ROOT)" fetch --depth 1 origin $(VCPKG_COMMIT)
	git -C "$(VCPKG_ROOT)" checkout --quiet $(VCPKG_COMMIT)
	"$(VCPKG_ROOT)/bootstrap-vcpkg.sh" -disableMetrics
	"$(VCPKG_ROOT)/vcpkg" install --triplet $(VCPKG_TRIPLET) --x-install-root="$(VCPKG_INSTALLED_DIR)"
