find_package(Python3 REQUIRED)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
)

add_lit_target(check-tensor-ops
  "Run all tensor-compiler tests"
  ${CMAKE_CURRENT_BINARY_DIR}
)
