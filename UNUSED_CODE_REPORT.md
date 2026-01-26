# Unused Code and Files Report

This report identifies code and files that are not used anywhere in the project.

## 🗑️ Completely Unused Files

### 1. `app/core/app_context.py`
- **Status**: Never imported or referenced
- **Content**: Contains commented-out code and unused imports
- **Recommendation**: Safe to delete

### 2. `app/cart/models.py`
- **Status**: File exists but class `CartItemView` is never used
- **Content**: Contains `CartItemView` dataclass (read-only representation)
- **Recommendation**: Either delete the file or remove the unused class if other code might be added later

### 3. `app/state_machine/handlers/system/__init__.py`
- **Status**: Empty directory with only an empty `__init__.py`
- **Content**: Empty file
- **Recommendation**: Safe to delete the entire directory

### 4. `app/tests/manual/test_menu_resolution.py`
- **Status**: Manual test file, not part of automated test suite
- **Content**: Standalone test script for menu resolution
- **Recommendation**: Keep if used for manual testing, otherwise consider removing or integrating into test suite

## ⚠️ Unused Classes/Functions

### 1. `CartItemView` in `app/cart/models.py`
- **Location**: `app/cart/models.py:8`
- **Status**: Defined but never imported or used
- **Recommendation**: Remove if not needed for future features

### 2. `WaitingForQuantityHandler` in `app/state_machine/handlers/item/add_item/waiting_for_quantity_handler.py`
- **Location**: `app/state_machine/handlers/item/add_item/waiting_for_quantity_handler.py:24`
- **Status**: Class is defined and the state `WAITING_FOR_QUANTITY` exists, but the handler is **NOT registered** in `TurnEngine.handlers` dictionary
- **Impact**: The state router references `WAITING_FOR_QUANTITY` state (line 59 in `state_router.py`), and uses the pattern `f"{state.name.lower()}_handler"` which would expect `waiting_for_quantity_handler`, but it's missing from the handler registry
- **Recommendation**: Either:
  - Register the handler in `app/core/turn_engine.py` (add to handlers dict), OR
  - Remove the handler file if quantity handling is done elsewhere

### 3. Duplicate Function in `app/state_machine/handlers/item/add_item/add_item_flow.py`
- **Location**: Lines 7-33 and 44-59
- **Status**: `determine_next_add_item_state()` is defined **twice** with different implementations
- **Impact**: The second definition (lines 44-59) overwrites the first one
- **Recommendation**: Remove one of the duplicate definitions (likely keep the more complete one)

## 🔴 Missing Handlers (Referenced but Not Implemented)

These handlers are referenced in the state router but not registered in `TurnEngine`:

### 1. `cancel_handler`
- **Location**: Referenced in `app/state_machine/state_router.py:204`
- **Status**: Handler name is returned but no handler registered in `turn_engine.py`
- **Impact**: Will cause "handler_not_implemented" response when cancel is triggered in non-IDLE states

### 2. `modifying_item_handler`
- **Location**: State `MODIFYING_ITEM` exists and is in router (line 66), but no handler registered
- **Status**: State exists but handler missing
- **Impact**: Will cause "handler_not_implemented" response

### 3. `removing_item_handler`
- **Location**: State `REMOVING_ITEM` exists and is in router (line 73), but no handler registered
- **Status**: State exists but handler missing
- **Impact**: Will cause "handler_not_implemented" response

## 📊 Summary

- **Unused Files**: 4 files
- **Unused Classes**: 2 classes
- **Missing Handlers**: 3 handlers (referenced but not implemented)
- **Code Issues**: 1 duplicate function definition

## 🎯 Recommended Actions

### High Priority (Breaking Issues)
1. **Fix missing handlers**: Either implement `cancel_handler`, `modifying_item_handler`, and `removing_item_handler`, or remove their references from the state router
2. **Fix `WaitingForQuantityHandler`**: Either register it in `turn_engine.py` or remove it if not needed
3. **Fix duplicate function**: Remove one of the duplicate `determine_next_add_item_state` definitions

### Low Priority (Cleanup)
1. Delete `app/core/app_context.py` (unused)
2. Delete or clean up `app/cart/models.py` (unused `CartItemView`)
3. Delete `app/state_machine/handlers/system/` directory (empty)
