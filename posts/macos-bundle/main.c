#include <Availability.h>
#include <mach-o/dyld.h>
#include <stdbool.h>
#include <stdio.h>

typedef bool (*CheckFunc)();

int main() {
// these APIs are only available on Mac OS X - not iPhone OS
#if __MAC_OS_X_VERSION_MIN_REQUIRED
    NSObjectFileImage ofi;
    if (NSCreateObjectFileImageFromFile("test.bundle", &ofi) !=
        NSObjectFileImageSuccess) {
        // FAIL("NSCreateObjectFileImageFromFile failed");
        return 1;
    }

    NSModule mod = NSLinkModule(ofi, "test.bundle", NSLINKMODULE_OPTION_NONE);
    if (mod == NULL) {
        // FAIL("NSLinkModule failed");
        return 1;
    }

    NSSymbol sym = NSLookupSymbolInModule(mod, "_checkdata");
    if (sym == NULL) {
        // FAIL("NSLookupSymbolInModule failed");
        return 1;
    }

    CheckFunc func = NSAddressOfSymbol(sym);
    if (!func()) {
        // FAIL("NSAddressOfSymbol failed");
        return 1;
    }

    if (!NSUnLinkModule(mod, NSUNLINKMODULE_OPTION_NONE)) {
        // FAIL("NSUnLinkModule failed");
        return 1;
    }

    if (!NSDestroyObjectFileImage(ofi)) {
        // FAIL("NSDestroyObjectFileImage failed");
        return 1;
    }
#endif
    printf("funck\n");
    // PASS("bundle-basic");
    return 0;
}
