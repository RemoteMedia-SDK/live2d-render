/* bindgen entry point.
 *
 * The Cubism Core SDK ships a single public header at
 * `Core/include/Live2DCubismCore.h`. We expose every `csm*` symbol it
 * declares to Rust via this wrapper. Build.rs sets `-IPATH/Core/include`
 * on the bindgen invocation. */

#include <Live2DCubismCore.h>
