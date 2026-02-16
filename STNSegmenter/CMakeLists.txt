#-----------------------------------------------------------------------------
set(MODULE_NAME STNSegmenter)

#-----------------------------------------------------------------------------
set(MODULE_PYTHON_SCRIPTS
  ${MODULE_NAME}.py
  )

set(MODULE_PYTHON_RESOURCES
  Resources/Icons/${MODULE_NAME}.png
  Resources/UI/${MODULE_NAME}.ui
  Resources/MNI/MNI152_T1_1mm_brain.nii.gz
  Resources/MNI/MNI152_T1_1mm_subbr_mask.nii.gz
  # nets
  Resources/nets/3.obj
  Resources/nets/cent_detect_hist.pkl
  Resources/nets/cent_pred.pt
  Resources/nets/mer_pca_mesh.pkl
  Resources/nets/shape_hist.pkl
  Resources/nets/shape_pcas.pkl
  Resources/nets/stn_shape_mask.pkl
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

  # Register the unittest subclass in the main script as a ctest.
  # Note that the test will also be available at runtime.
  slicer_add_python_unittest(SCRIPT ${MODULE_NAME}.py)

  # Additional build-time testing
  add_subdirectory(Testing)
endif()
