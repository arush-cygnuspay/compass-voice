# app/core/app_context.py

from app.cart.cart import Cart
from app.menu.repository import MenuRepository
from app.state_machine.handlers.item.confirming_handler import ConfirmingHandler
from app.state_machine.handlers.item.idle_handler import IdleHandler

cart = Cart()

# handlers = {
#     "idle_handler": IdleHandler(menu_repo),
#     "confirming_handler": ConfirmingHandler(menu_repo, cart),
# }
