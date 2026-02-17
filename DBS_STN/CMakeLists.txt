#-----------------------------------------------------------------------------
set(MODULE_NAME DBSShiftPrediction)

#-----------------------------------------------------------------------------
# Install the main scripted module and your package files (preserving paths)
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py

  Lib/__init__.py
  Lib/mer_support.py
  Lib/slicer_preprocessing.py
  Lib/image_utils.py
  Lib/utils_file.py
)

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  Resources/nets/net_transformer.pt
  Resources/nets/mer_pca_mesh.pkl
  Resources/nets/transformer_classifier_test.pt
)

#-----------------------------------------------------------------------------
slicerMacroBuildScriptedModule(
  NAME ${MODULE_NAME}
  SCRIPTS ${MODULE_PYTHON_SCRIPTS}
  RESOURCES ${MODULE_PYTHON_RESOURCES}
  WITH_GENERIC_TESTS
)

#-----------------------------------------------------------------------------
if(BUILD_TESTING)
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)
  add_subdirectory(Testing)
endif()
