idf_component_register(
    SRCS "main.cpp" "board.c" "ui.c" "touch_calib.c"
         "cmlp_model.cpp" "generate.cpp" "engine.cpp" "self_test.cpp"
         "sampler.cpp" "vocab.cpp" "vocab_encode.cpp"
    INCLUDE_DIRS "."
    REQUIRES esp_lcd driver nvs_flash esp_partition esp_timer)

target_compile_options(${COMPONENT_LIB} PRIVATE -O2)

# `idf.py flash` also writes the model into the "model" partition.
# `idf.py app-flash` leaves it alone (handy when only the code changes).
set(MODEL_BIN "${PROJECT_DIR}/model/model.bin")
if(EXISTS "${MODEL_BIN}")
    esptool_py_flash_to_partition(flash "model" "${MODEL_BIN}")
else()
    message(WARNING "model/model.bin not found: 'idf.py flash' will not write the model partition")
endif()
