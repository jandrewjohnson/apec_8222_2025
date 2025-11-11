import random

# --- Product Class ---
class Product:
    """Represents a generic product with a name and a base value."""
    def __init__(self, name, base_value):
        self.name = name
        self.base_value = float(base_value) # Intrinsic or production value

    def __str__(self):
        return f"{self.name} (Base Value: ${self.base_value:.2f})"

# --- Seller Class ---
class Seller:
    """
    Represents a seller who stocks products and sets prices.
    """
    def __init__(self, name, markup_percentage=20.0):
        self.name = name
        # Inventory: {product_name: {'product_object': Product, 'quantity': int, 'cost_price': float}}
        self.inventory = {}
        self.markup_percentage = float(markup_percentage)
        self.cash_earned = 0.0
        print(f"Seller '{self.name}' is open for business with a {self.markup_percentage}% markup.")

    def stock_product(self, product_object, quantity, cost_price=None):
        """Adds a product to the seller's inventory."""
        if cost_price is None:
            cost_price = product_object.base_value # Seller's cost is the product's base value by default
        
        if product_object.name in self.inventory:
            self.inventory[product_object.name]['quantity'] += quantity
        else:
            self.inventory[product_object.name] = {
                'product_object': product_object,
                'quantity': quantity,
                'cost_price': float(cost_price) # The price seller paid or values it at
            }
        print(f"Seller '{self.name}' stocked {quantity} of '{product_object.name}' at a cost of ${cost_price:.2f} each.")

    def get_selling_price(self, product_name):
        """Calculates the selling price for a product based on cost and markup."""
        if product_name in self.inventory:
            item_details = self.inventory[product_name]
            cost = item_details['cost_price']
            price = cost * (1 + self.markup_percentage / 100.0)
            return round(price, 2)
        return None

    def make_sale(self, product_name, quantity_requested):
        """Processes a sale if the product is in stock."""
        if product_name not in self.inventory or self.inventory[product_name]['quantity'] == 0:
            # print(f"Seller '{self.name}': Sorry, '{product_name}' is out of stock.")
            return False, 0 # Sale failed, price is 0

        item_details = self.inventory[product_name]
        
        if quantity_requested > item_details['quantity']:
            # print(f"Seller '{self.name}': Not enough '{product_name}' in stock. Only {item_details['quantity']} available.")
            return False, 0 # Sale failed for requested quantity

        selling_price_per_unit = self.get_selling_price(product_name)
        total_sale_value = selling_price_per_unit * quantity_requested

        item_details['quantity'] -= quantity_requested
        self.cash_earned += total_sale_value
        
        print(f"Seller '{self.name}' sold {quantity_requested} of '{product_name}' for ${total_sale_value:.2f}.")
        return True, selling_price_per_unit # Sale successful, return price per unit

    def display_wares(self):
        print(f"\n--- Wares from Seller '{self.name}' ---")
        if not self.inventory:
            print("Currently no items in stock.")
            return
        for name, details in self.inventory.items():
            if details['quantity'] > 0:
                price = self.get_selling_price(name)
                print(f"- {details['product_object'].name}: {details['quantity']} available @ ${price:.2f} each")
        print("------------------------------------")

    def __str__(self):
        return f"Seller: {self.name}, Cash Earned: ${self.cash_earned:.2f}"


# --- Buyer Class ---
class Buyer:
    """
    Represents a buyer with a budget and preferences for products.
    """
    def __init__(self, name, budget, preferences=None):
        """
        preferences: dict, e.g., {'Apples': 1.2} means buyer values apples 20% more than base value.
                     A value of 1.0 means they value it at its base value.
                     A value < 1.0 means they undervalue it.
        """
        self.name = name
        self.budget = float(budget)
        # Preferences: product_name -> willingness_to_pay_multiplier
        self.preferences = preferences if preferences is not None else {}
        self.items_owned = {} # product_name -> quantity
        print(f"Buyer '{self.name}' has entered the market with a budget of ${self.budget:.2f}.")

    def _perceived_value(self, product_object):
        """Calculates how much the buyer values a product based on their preferences."""
        multiplier = self.preferences.get(product_object.name, 0.8) # Default to valuing less if no preference
        return product_object.base_value * multiplier

    def consider_purchase(self, product_object, seller_price, seller_name):
        """Buyer considers if they want to buy a product at a given price."""
        perceived_value = self._perceived_value(product_object)
        
        print(f"Buyer '{self.name}' considering '{product_object.name}' from '{seller_name}': "
              f"Price ${seller_price:.2f}, Perceived Value ${perceived_value:.2f}")

        if seller_price <= self.budget and seller_price <= perceived_value:
            # Basic decision: Buy 1 unit if affordable and price is <= perceived value
            print(f"  Decision: '{self.name}' wants to buy '{product_object.name}' at ${seller_price:.2f}.")
            return True, 1 # Wants to buy, quantity 1
        elif seller_price > perceived_value:
            print(f"  Decision: Too expensive for '{self.name}' (values it less).")
        elif seller_price > self.budget:
            print(f"  Decision: '{self.name}' cannot afford '{product_object.name}' at this price.")
        return False, 0 # Does not want to buy

    def attempt_to_buy_from_seller(self, seller_object):
        """Buyer looks at a seller's wares and tries to buy things."""
        print(f"\nBuyer '{self.name}' (Budget: ${self.budget:.2f}) is browsing at '{seller_object.name}'s shop.")
        items_bought_this_visit = 0
        for product_name, details in seller_object.inventory.items():
            if details['quantity'] > 0: # Only consider items in stock
                product_obj = details['product_object']
                seller_price = seller_object.get_selling_price(product_name)
                
                wants_to_buy, quantity_to_buy = self.consider_purchase(product_obj, seller_price, seller_object.name)
                
                if wants_to_buy:
                    # Try to make the actual purchase from the seller
                    sale_successful, price_paid_per_unit = seller_object.make_sale(product_name, quantity_to_buy)
                    
                    if sale_successful:
                        total_cost = price_paid_per_unit * quantity_to_buy
                        self.budget -= total_cost
                        self.items_owned[product_name] = self.items_owned.get(product_name, 0) + quantity_to_buy
                        print(f"  Buyer '{self.name}' successfully bought {quantity_to_buy} '{product_name}'. Budget left: ${self.budget:.2f}")
                        items_bought_this_visit += 1
                    else:
                        print(f"  Buyer '{self.name}': Purchase of '{product_name}' failed (e.g., out of stock by now).")
        if items_bought_this_visit == 0:
            print(f"Buyer '{self.name}' did not buy anything from '{seller_object.name}' this visit.")


    def display_possessions(self):
        print(f"\n--- Possessions of Buyer '{self.name}' ---")
        if not self.items_owned:
            print("Nothing owned yet.")
            return
        for name, quantity in self.items_owned.items():
            print(f"- {quantity} of {name}")
        print(f"Budget remaining: ${self.budget:.2f}")
        print("------------------------------------")

    def __str__(self):
        return f"Buyer: {self.name}, Budget: ${self.budget:.2f}"

# --- Market Simulation ---
if __name__ == "__main__":
    print("--- Welcome to the Mini-Market Simulation! ---")

    # 1. Create Products
    apple = Product(name="Apple", base_value=0.50)
    bread = Product(name="Bread Loaf", base_value=2.00)
    cheese = Product(name="Cheese Block", base_value=4.00)
    fancy_pen = Product(name="Fancy Pen", base_value=10.00)

    # 2. Create Sellers and Stock Products
    seller_alice = Seller(name="Alice's Goods", markup_percentage=30)
    seller_alice.stock_product(apple, quantity=50)
    seller_alice.stock_product(bread, quantity=20)
    seller_alice.stock_product(fancy_pen, quantity=5, cost_price=8.00) # Alice got pens cheaper

    seller_bob = Seller(name="Bob's Emporium", markup_percentage=25)
    seller_bob.stock_product(apple, quantity=30, cost_price=0.45) # Bob's apple cost
    seller_bob.stock_product(cheese, quantity=15)
    seller_bob.stock_product(bread, quantity=25) # Bob has more bread initially
    seller_bob.stock_product(fancy_pen, quantity=10)


    # 3. Create Buyers with Budgets and Preferences
    buyer_charlie = Buyer(name="Charlie", budget=20.00, preferences={"Apple": 1.3, "Bread Loaf": 1.1, "Cheese Block": 0.9})
    buyer_diana = Buyer(name="Diana", budget=15.00, preferences={"Cheese Block": 1.5, "Fancy Pen": 1.2, "Apple": 1.0})
    buyer_edward = Buyer(name="Edward", budget=30.00, preferences={"Bread Loaf": 1.2, "Fancy Pen": 0.8}) # Edward values pens less

    all_sellers = [seller_alice, seller_bob]
    all_buyers = [buyer_charlie, buyer_diana, buyer_edward]

    # 4. Simulate Market Rounds
    num_rounds = 2
    for round_num in range(1, num_rounds + 1):
        print(f"\n\n--- MARKET ROUND {round_num} ---")
        # Randomize seller order for fairness, or buyer order
        random.shuffle(all_sellers)
        
        for buyer in all_buyers:
            if buyer.budget <= 0:
                print(f"\nBuyer '{buyer.name}' has no budget left, skipping their turn.")
                continue
            
            # Buyer chooses a seller (could be random, or they browse all)
            # For simplicity, let's have them browse one random seller per round or cycle through
            seller_to_visit = random.choice(all_sellers) 
            # Alternatively, to visit all:
            # for seller_to_visit in all_sellers:
            #     buyer.attempt_to_buy_from_seller(seller_to_visit)
            #     if buyer.budget <=0: break # Stop if budget runs out mid-browsing
            
            buyer.attempt_to_buy_from_seller(seller_to_visit)


        print("\n--- End of Round Summary ---")
        for seller in all_sellers:
            print(seller)
            seller.display_wares() # Show remaining stock
        for buyer in all_buyers:
            buyer.display_possessions()


    print("\n\n--- FINAL MARKET STATE ---")
    for seller in all_sellers:
        print(seller)
    for buyer in all_buyers:
        print(buyer)
        buyer.display_possessions()