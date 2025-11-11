import random
import math # For utility functions if needed (e.g., log)

# --- Product Class (can remain largely the same) ---
class Product:
    def __init__(self, name, base_value=1.0): # Base value might be less relevant now
        self.name = name
        self.base_value = float(base_value) # Could represent societal value or a benchmark

    def __str__(self):
        return f"{self.name}"

# --- Producer Class ---
class Producer:
    """
    Represents a producer who uses inputs to create products and sells them.
    """
    def __init__(self, name, product_to_produce, production_coeffs, input_costs):
        """
        Args:
            product_to_produce (Product): The product object this producer makes.
            production_coeffs (dict): Coefficients for a simple production function.
                                      e.g., {'labor': 0.5, 'capital': 0.5} for Q = L^0.5 * K^0.5
                                      For simplicity, we'll use Q = coeff * Input (linear for one input)
                                      Or more generally, Q = sum(coeff_i * input_i) if multiple inputs.
                                      Let's use a simplified: Q = input_units * productivity_factor
            input_costs (dict): Cost per unit of each input. e.g., {'labor': 10}
        """
        self.name = name
        self.product_produced = product_to_produce
        self.production_coeffs = production_coeffs # e.g., {'labor_productivity': 2} -> 1 unit labor produces 2 products
        self.input_costs = input_costs          # e.g., {'labor': 5} cost per unit of labor
        
        self.inventory = {product_to_produce.name: 0} # Quantity of produced goods
        self.cash_balance = 100.0 # Starting cash to buy inputs
        self.total_units_produced = 0
        self.total_revenue = 0.0
        self.total_production_cost = 0.0
        
        print(f"Producer '{self.name}' established, producing '{self.product_produced.name}'.")

    def decide_production_level(self, market_price_estimate=None):
        """
        Simple heuristic: Produce if cash allows and there's potential demand (or fixed amount).
        A more complex model would use profit maximization (marginal revenue = marginal cost).
        For now, let's say they decide to use a certain amount of one primary input if they can afford it.
        """
        # Assuming one primary input for simplicity, e.g., 'labor'
        primary_input_name = list(self.production_coeffs.keys())[0].replace('_productivity', '') # e.g. 'labor'
        input_productivity = self.production_coeffs.get(f'{primary_input_name}_productivity', 1)
        cost_per_input_unit = self.input_costs.get(primary_input_name, float('inf'))

        # Decide how many units of input to use (e.g., up to 10 if affordable)
        max_input_units_to_consider = 10
        input_units_to_use = 0
        for units in range(max_input_units_to_consider, 0, -1):
            if self.cash_balance >= (units * cost_per_input_unit):
                input_units_to_use = units
                break
        
        if input_units_to_use == 0:
            print(f"Producer '{self.name}' cannot afford any {primary_input_name} input units right now.")
            return 0
            
        return input_units_to_use # This is units of input, not product

    def produce(self, input_units_used):
        """Produces goods based on input units and production function."""
        if input_units_used <= 0:
            return

        primary_input_name = list(self.production_coeffs.keys())[0].replace('_productivity', '')
        input_productivity = self.production_coeffs.get(f'{primary_input_name}_productivity', 1)
        cost_per_input_unit = self.input_costs.get(primary_input_name, 0)

        cost_of_inputs = input_units_used * cost_per_input_unit
        
        if self.cash_balance < cost_of_inputs:
            print(f"Producer '{self.name}' cannot afford to produce (cost ${cost_of_inputs:.2f}, balance ${self.cash_balance:.2f}).")
            return

        self.cash_balance -= cost_of_inputs
        self.total_production_cost += cost_of_inputs
        
        quantity_produced = input_units_used * input_productivity # Linear production
        
        self.inventory[self.product_produced.name] += quantity_produced
        self.total_units_produced += quantity_produced
        print(f"Producer '{self.name}' used {input_units_used} units of {primary_input_name}, spent ${cost_of_inputs:.2f}, "
              f"and produced {quantity_produced} of '{self.product_produced.name}'. "
              f"Inventory: {self.inventory[self.product_produced.name]}. Cash: ${self.cash_balance:.2f}")

    def set_price(self, base_markup=0.2):
        """
        Sets a selling price. Could be based on marginal cost, average cost, or market conditions.
        Simple: Average cost of production for this batch + markup.
        If no production this round, could use last known cost or a default.
        For simplicity, let's use (total_production_cost / total_units_produced) * (1 + markup) if possible,
        otherwise a fallback based on input costs.
        """
        if self.total_units_produced > 0:
            avg_cost = self.total_production_cost / self.total_units_produced
        else: # Fallback if nothing produced yet
            primary_input_name = list(self.production_coeffs.keys())[0].replace('_productivity', '')
            input_productivity = self.production_coeffs.get(f'{primary_input_name}_productivity', 1)
            cost_per_input_unit = self.input_costs.get(primary_input_name, 1)
            if input_productivity == 0: return float('inf') # Cannot produce
            avg_cost = cost_per_input_unit / input_productivity # Cost per unit of product

        price = avg_cost * (1 + base_markup)
        # Price floor based on input costs for 1 unit of product
        min_price = (self.input_costs.get(list(self.input_costs.keys())[0], 1) / 
                     self.production_coeffs.get(list(self.production_coeffs.keys())[0],1) )*1.05 # Min 5% markup on direct input cost
        
        return max(round(price, 2), round(min_price,2))


    def make_sale(self, quantity_requested, price_per_unit):
        """Sells products if available."""
        product_name = self.product_produced.name
        if self.inventory.get(product_name, 0) >= quantity_requested:
            self.inventory[product_name] -= quantity_requested
            revenue = quantity_requested * price_per_unit
            self.cash_balance += revenue
            self.total_revenue += revenue
            print(f"Producer '{self.name}' sold {quantity_requested} of '{product_name}' for ${revenue:.2f}. "
                  f"Cash: ${self.cash_balance:.2f}")
            return True
        # print(f"Producer '{self.name}': Not enough '{product_name}' in stock for sale ({self.inventory.get(product_name,0)} avail).")
        return False
        
    def __str__(self):
        profit = self.total_revenue - self.total_production_cost
        return (f"Producer: {self.name} (Produces: {self.product_produced.name}), "
                f"Inventory: {self.inventory.get(self.product_produced.name, 0)}, "
                f"Cash: ${self.cash_balance:.2f}, Total Profit: ${profit:.2f}")

# --- Consumer Class ---
class Consumer:
    """
    Represents a consumer with a budget and a utility function.
    Aims to maximize utility given their budget.
    """
    def __init__(self, name, budget, utility_params):
        """
        Args:
            utility_params (dict): Parameters for a utility function.
                                   e.g., {'product_name': {'alpha': 0.5, 'saturation': 10}}
                                   Utility for product X = alpha * log(1 + quantity_X) (diminishing marginal utility)
                                   Or simpler: U = alpha * quantity (linear, no diminishing MU for simplicity here)
                                   Let's use: U_product = alpha * quantity - 0.05 * beta * quantity^2 (diminishing MU)
        """
        self.name = name
        self.budget_initial = float(budget)
        self.budget_remaining = float(budget)
        self.utility_params = utility_params # e.g. {'Apples': {'alpha': 5, 'beta': 0.5}}
        self.items_consumed = {} # product_name -> quantity consumed
        self.total_utility = 0.0
        print(f"Consumer '{self.name}' enters with budget ${self.budget_remaining:.2f}.")

    def get_marginal_utility(self, product_name, current_quantity_consumed):
        """Calculates the marginal utility of consuming one more unit of a product."""
        params = self.utility_params.get(product_name)
        if not params:
            return 0 # No utility from this product

        alpha = params.get('alpha', 1)
        beta = params.get('beta', 0.1) # Coefficient for diminishing returns

        # MU = dU/dQ = alpha - beta * Q (if U = alpha*Q - 0.5*beta*Q^2)
        # So, MU for the (Q+1)th unit is approximately U(Q+1) - U(Q)
        # U(Q) = alpha * Q - 0.05 * beta * Q**2
        # U(Q+1) = alpha * (Q+1) - 0.05 * beta * (Q+1)**2
        # MU_next = alpha - 0.05 * beta * (2*Q + 1)
        
        mu = alpha - beta * current_quantity_consumed 
        return max(0, mu) # Marginal utility cannot be negative

    def decide_purchases(self, available_producers):
        """
        Consumer decides what to buy from available producers to maximize utility.
        Simple greedy approach: keep buying the item that gives the highest
        marginal utility per dollar (MU/P) until budget runs out or MU/P <= 0.
        """
        print(f"\nConsumer '{self.name}' (Budget: ${self.budget_remaining:.2f}) is shopping...")
        
        purchases_made_this_round = False
        
        # Loop until no more beneficial purchases can be made or budget exhausted
        while True:
            best_mu_per_dollar = -1.0
            best_item_to_buy = None
            producer_of_best_item = None
            price_of_best_item = 0

            for producer in available_producers:
                product_name = producer.product_produced.name
                if producer.inventory.get(product_name, 0) > 0: # If producer has stock
                    price = producer.set_price() # Producer sets current price
                    
                    if price <= 0 or price == float('inf') : continue # Skip if price is invalid

                    current_consumed_quantity = self.items_consumed.get(product_name, 0)
                    mu = self.get_marginal_utility(product_name, current_consumed_quantity)
                    mu_per_dollar = mu / price if price > 0 else float('-inf') # Avoid division by zero

                    if mu_per_dollar > best_mu_per_dollar and self.budget_remaining >= price:
                        best_mu_per_dollar = mu_per_dollar
                        best_item_to_buy = product_name
                        producer_of_best_item = producer
                        price_of_best_item = price
            
            if best_item_to_buy and best_mu_per_dollar > 0: # Buy if MU/P is positive and item found
                print(f"  '{self.name}' considers buying '{best_item_to_buy}' from '{producer_of_best_item.name}' "
                      f"at ${price_of_best_item:.2f} (MU/P = {best_mu_per_dollar:.2f})")
                
                if producer_of_best_item.make_sale(quantity_requested=1, price_per_unit=price_of_best_item):
                    self.budget_remaining -= price_of_best_item
                    
                    # Calculate utility from this one unit
                    current_q = self.items_consumed.get(best_item_to_buy, 0)
                    utility_gain = self.get_marginal_utility(best_item_to_buy, current_q) # MU of this specific unit
                    self.total_utility += utility_gain 
                    
                    self.items_consumed[best_item_to_buy] = current_q + 1
                    
                    print(f"    '{self.name}' bought 1 '{best_item_to_buy}'. "
                          f"Utility gain: {utility_gain:.2f}. Total Utility: {self.total_utility:.2f}. Budget: ${self.budget_remaining:.2f}")
                    purchases_made_this_round = True
                else:
                    # print(f"    Purchase of '{best_item_to_buy}' failed (e.g. producer ran out just now).")
                    break # Stop if a desired purchase fails to avoid infinite loop on same item
            else:
                # No more beneficial items to buy or budget exhausted for best items
                if not purchases_made_this_round and best_mu_per_dollar <= 0 :
                    print(f"  '{self.name}' finds no more items offering positive marginal utility per dollar within budget.")
                elif not purchases_made_this_round :
                     print(f"  '{self.name}' made no purchases this round.")
                break 
        
    def __str__(self):
        return (f"Consumer: {self.name}, Budget Left: ${self.budget_remaining:.2f}, "
                f"Total Utility: {self.total_utility:.2f}, Consumed: {self.items_consumed}")

# --- Market Simulation ---
if __name__ == "__main__":
    print("--- Advanced Market Simulation: Producers & Consumers ---")

    # 1. Define Products
    food = Product(name="FoodUnit")
    tool = Product(name="ToolUnit")

    # 2. Create Producers
    # Producer(name, product_to_produce, production_coeffs={'input_productivity': X}, input_costs={'input_name': Y})
    farm_alpha = Producer(name="FarmAlpha", product_to_produce=food,
                          production_coeffs={'labor_productivity': 2}, # 1 labor -> 2 food
                          input_costs={'labor': 5}) # Cost of 1 labor unit is 5
    
    factory_beta = Producer(name="FactoryBeta", product_to_produce=tool,
                            production_coeffs={'parts_productivity': 1}, # 1 parts_input -> 1 tool
                            input_costs={'parts': 10}) # Cost of 1 parts_input is 10

    all_producers = [farm_alpha, factory_beta]

    # 3. Create Consumers
    # Consumer(name, budget, utility_params={'Product': {'alpha': A, 'beta': B}})
    # U = Alpha*Q - 0.5*Beta*Q^2  => MU = Alpha - Beta*Q
    consumer_eve = Consumer(name="Eve", budget=100, 
                            utility_params={
                                "FoodUnit": {'alpha': 10, 'beta': 1.0}, # High initial desire, quick diminishing MU for food
                                "ToolUnit": {'alpha': 25, 'beta': 5.0}  # Higher desire for tools, slower diminishing MU
                            })
    consumer_adam = Consumer(name="Adam", budget=80,
                             utility_params={
                                 "FoodUnit": {'alpha': 12, 'beta': 1.2},
                                 "ToolUnit": {'alpha': 15, 'beta': 3.0}
                             })
    all_consumers = [consumer_eve, consumer_adam]

    # 4. Simulate Market Rounds
    num_rounds = 3
    for round_num in range(1, num_rounds + 1):
        print(f"\n\n--- MARKET ROUND {round_num} ---")

        # a. Producers decide production levels and produce
        print("\n-- Production Phase --")
        for producer in all_producers:
            # For simplicity, assume they can estimate a price or just produce a set amount
            input_units = producer.decide_production_level() 
            producer.produce(input_units)
        
        print("\n-- Current Producer Offerings --")
        for producer in all_producers:
             if producer.inventory.get(producer.product_produced.name, 0) > 0:
                print(f"  {producer.name} offers {producer.product_produced.name} at ${producer.set_price():.2f} "
                      f"(Stock: {producer.inventory[producer.product_produced.name]})")


        # b. Consumers make purchasing decisions
        print("\n-- Consumption Phase --")
        random.shuffle(all_consumers) # Randomize consumer order
        for consumer in all_consumers:
            if consumer.budget_remaining > 0:
                consumer.decide_purchases(all_producers)
            else:
                print(f"Consumer '{consumer.name}' has no budget.")
        
        print("\n--- End of Round Summary ---")
        for producer in all_producers:
            print(producer)
        for consumer in all_consumers:
            print(consumer)

    print("\n\n--- FINAL MARKET STATE ---")
    for producer in all_producers:
        print(producer)
    for consumer in all_consumers:
        print(consumer)