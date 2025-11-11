import numpy as np
import matplotlib.pyplot as plt
import random

# --- Product Class ---
class Product:
    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

# --- Producer Class ---
class Producer:
    def __init__(self, name, product, base_cost_per_unit, cost_slope, max_capacity=100):
        """
        Args:
            base_cost_per_unit (float): The cost of producing the very first unit (intercept of MC).
            cost_slope (float): How much marginal cost increases for each additional unit produced.
                                MC(q) = base_cost_per_unit + cost_slope * q
            max_capacity (int): Maximum units this producer can make.
        """
        self.name = name
        self.product = product
        self.base_cost_per_unit = float(base_cost_per_unit)
        self.cost_slope = float(cost_slope) # For increasing marginal cost
        self.max_capacity = int(max_capacity)
        # print(f"Producer '{self.name}' for '{product.name}' established.")

    def marginal_cost(self, quantity_already_produced):
        """Calculates the cost of producing the *next* unit."""
        # MC(q) = d(TotalCost)/dq.
        # If TC(q) = base_cost*q + 0.5*slope*q^2, then MC(q) = base_cost + slope*q
        # This is the cost for the (q+1)th unit.
        if quantity_already_produced >= self.max_capacity:
            return float('inf') # Cannot produce more
        mc = self.base_cost_per_unit + self.cost_slope * quantity_already_produced
        return mc

    def quantity_supplied_at_price(self, price):
        """Determines how much this producer will supply at a given market price."""
        if price < self.base_cost_per_unit: # If price doesn't even cover the first unit's MC
            return 0
        
        quantity_to_supply = 0
        # Producer supplies as long as Price >= Marginal Cost for that unit
        for q_produced in range(self.max_capacity + 1): # Test from 0 up to capacity
            mc_for_next_unit = self.marginal_cost(q_produced) # MC for the (q_produced+1)th unit
            if price >= mc_for_next_unit:
                if q_produced < self.max_capacity : # If they can produce one more
                     quantity_to_supply = q_produced + 1
                else: # They are at capacity, and price covers MC of last unit
                    quantity_to_supply = self.max_capacity
                    break # can't produce more
            else: # Price is less than MC for the next unit
                break # Stop supplying more
        return quantity_to_supply

    def __str__(self):
        return f"Producer: {self.name} (Product: {self.product.name})"


# --- Consumer Class ---
class Consumer:
    def __init__(self, name, product, max_value_first_unit, value_slope, budget=1000):
        """
        Args:
            max_value_first_unit (float): Max price consumer is willing to pay for the first unit.
            value_slope (float): How much their marginal valuation decreases for each additional unit.
                                 MU(q) = max_value_first_unit - value_slope * q (diminishing MU)
            budget (float): Total budget (less critical for generating demand curve, more for actual purchase).
        """
        self.name = name
        self.product = product
        self.max_value_first_unit = float(max_value_first_unit)
        self.value_slope = float(value_slope) # For diminishing marginal utility
        self.budget = float(budget) # We'll consider this for affordability
        # print(f"Consumer '{self.name}' interested in '{product.name}' established.")

    def marginal_utility_value(self, quantity_already_consumed):
        """
        Calculates the monetary value (willingness to pay) for the *next* unit.
        This is their marginal utility expressed in dollars.
        MU(q) for the (q+1)th unit.
        """
        mu = self.max_value_first_unit - self.value_slope * quantity_already_consumed
        return max(0, mu) # Cannot have negative willingness to pay

    def quantity_demanded_at_price(self, price):
        """Determines how much this consumer will demand at a given market price."""
        if price > self.max_value_first_unit: # If price is higher than value of first unit
            return 0
            
        quantity_to_demand = 0
        cumulative_cost = 0
        # Consumer demands as long as Marginal Utility (value) >= Price for that unit
        # and they can afford it (considering cumulative cost).
        for q_consumed in range(1000): # Arbitrary large number for potential consumption
            mu_for_next_unit = self.marginal_utility_value(q_consumed) # MU for (q_consumed+1)th unit
            
            if mu_for_next_unit >= price:
                if cumulative_cost + price <= self.budget:
                    quantity_to_demand = q_consumed + 1
                    cumulative_cost += price
                else: # Cannot afford the next unit even if MU is high enough
                    break
            else: # Marginal utility is less than price
                break
        return quantity_to_demand
        
    def __str__(self):
        return f"Consumer: {self.name} (Interested in: {self.product.name})"

# --- Market Simulation and Equilibrium ---
def simulate_market(producers, consumers, product, price_range):
    """
    Simulates the market to generate supply and demand schedules.
    Args:
        producers (list): List of Producer objects.
        consumers (list): List of Consumer objects.
        product (Product): The product being traded.
        price_range (np.array): Array of prices to test.
    Returns:
        tuple: (demand_schedule, supply_schedule)
               Each schedule is a list of (price, total_quantity) tuples.
    """
    demand_schedule_points = [] # (price, total_quantity_demanded)
    supply_schedule_points = [] # (price, total_quantity_supplied)

    print(f"\n--- Simulating Market for '{product.name}' ---")
    for price in price_range:
        total_q_demanded_at_price = 0
        for consumer in consumers:
            if consumer.product.name == product.name: # Ensure consumer is interested in this product
                total_q_demanded_at_price += consumer.quantity_demanded_at_price(price)
        
        total_q_supplied_at_price = 0
        for producer in producers:
            if producer.product.name == product.name: # Ensure producer makes this product
                total_q_supplied_at_price += producer.quantity_supplied_at_price(price)
        
        demand_schedule_points.append((price, total_q_demanded_at_price))
        supply_schedule_points.append((price, total_q_supplied_at_price))
        # print(f"Price: ${price:.2f} | Qd: {total_q_demanded_at_price} | Qs: {total_q_supplied_at_price}")

    return demand_schedule_points, supply_schedule_points


def find_equilibrium(demand_schedule_points, supply_schedule_points):
    """
    Finds the approximate equilibrium price and quantity.
    This version finds the point where |Qd - Qs| is minimized.
    A more advanced version would interpolate for exact intersection.
    """
    prices_d = np.array([p[0] for p in demand_schedule_points])
    q_demanded = np.array([p[1] for p in demand_schedule_points])
    
    prices_s = np.array([p[0] for p in supply_schedule_points]) # Should be same as prices_d
    q_supplied = np.array([p[1] for p in supply_schedule_points])

    # For simplicity, let's find where Qd and Qs are closest or cross
    # This assumes prices are sorted and consistent between schedules
    
    equilibrium_p = None
    equilibrium_q = None
    min_diff = float('inf')
    
    # Iterate to find where demand quantity is greater than supply quantity,
    # and then flips, or where they are closest.
    prev_q_d = -1
    prev_q_s = -1

    for i in range(len(prices_d)):
        p, qd, qs = prices_d[i], q_demanded[i], q_supplied[i]
        
        diff = abs(qd - qs)
        if diff < min_diff:
            min_diff = diff
            equilibrium_p = p
            equilibrium_q = (qd + qs) / 2 # Average q at this price

        # Check for crossing point (more robust for discrete steps)
        if i > 0:
            # Demand crosses supply from above
            if prev_q_d >= prev_q_s and qd <= qs:
                # More precise would be to interpolate between (prices_d[i-1], prices_d[i])
                # For now, take the current point or the one where diff is minimal
                if diff <= min_diff : # Prioritize this if it's also the minimum difference
                    equilibrium_p = p
                    equilibrium_q = (qd + qs) / 2
                # If no exact match, this 'p' is where supply starts to meet or exceed demand
                # We could also say equilibrium is between prices_d[i-1] and prices_d[i]
                # print(f"Crossing detected between P={prices_d[i-1]:.2f} and P={p:.2f}")
                break # Found a crossing, good enough for this example

            # Supply crosses demand from above (less common for standard curves)
            elif prev_q_d <= prev_q_s and qd >= qs:
                 if diff <= min_diff :
                    equilibrium_p = p
                    equilibrium_q = (qd + qs) / 2
                 break
        
        prev_q_d, prev_q_s = qd, qs

    if equilibrium_p is None and len(prices_d)>0: # Fallback if no clear crossing found with simple logic
        idx_min_diff = np.argmin(np.abs(q_demanded - q_supplied))
        equilibrium_p = prices_d[idx_min_diff]
        equilibrium_q = (q_demanded[idx_min_diff] + q_supplied[idx_min_diff]) / 2
        print("Equilibrium approximated by minimum difference.")


    if equilibrium_p is not None:
        print(f"\nApproximate Equilibrium: Price=${equilibrium_p:.2f}, Quantity={equilibrium_q:.2f} (Diff Qd-Qs: {min_diff:.2f})")
    else:
        print("\nCould not determine a clear equilibrium point with this data.")
        
    return equilibrium_p, equilibrium_q


def plot_supply_demand(demand_schedule, supply_schedule, equilibrium_p, equilibrium_q, product_name):
    if not plt: # Matplotlib not available
        return

    prices_d = [p[0] for p in demand_schedule]
    q_demanded = [p[1] for p in demand_schedule]
    
    prices_s = [p[0] for p in supply_schedule]
    q_supplied = [p[1] for p in supply_schedule]

    plt.figure(figsize=(10, 6))
    plt.plot(q_demanded, prices_d, label='Demand Curve (Market)', color='blue', marker='o', linestyle='-')
    plt.plot(q_supplied, prices_s, label='Supply Curve (Market)', color='red', marker='x', linestyle='-')

    if equilibrium_p is not None and equilibrium_q is not None:
        plt.scatter([equilibrium_q], [equilibrium_p], color='green', s=100, zorder=5, label='Equilibrium')
        plt.axhline(y=equilibrium_p, color='gray', linestyle='--', xmax=(equilibrium_q / plt.xlim()[1] if plt.xlim()[1] > 0 else 1))
        plt.axvline(x=equilibrium_q, color='gray', linestyle='--', ymax=(equilibrium_p / plt.ylim()[1] if plt.ylim()[1] > 0 else 1))
        plt.text(equilibrium_q * 1.05, equilibrium_p * 1.05, f' E (P*=${equilibrium_p:.2f}, Q*={equilibrium_q:.0f})', color='green')


    plt.title(f'Supply and Demand for {product_name}')
    plt.xlabel('Quantity')
    plt.ylabel('Price ($)')
    plt.legend()
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.ylim(bottom=0) # Price cannot be negative
    plt.xlim(left=0)   # Quantity cannot be negative
    # Set vertical max to 12
    plt.ylim(top=12)
    # Set horizontal max t0 400
    plt.xlim(right=400)
    
    # Save the plot
    import os
    plot_filename = f"supply_demand_{product_name.lower().replace(' ', '_')}.png"
    IMAGE_OUTPUT_DIR = "images" # Assuming this is defined elsewhere
    os.makedirs(IMAGE_OUTPUT_DIR, exist_ok=True) # Create directory if it doesn't exist
    plt.savefig(os.path.join(IMAGE_OUTPUT_DIR, plot_filename))
    print(f"Plot saved as {plot_filename}")
    plt.show() # Display the plot


# --- Main Simulation ---
if __name__ == "__main__":
    print("--- Market Simulation: Supply, Demand, and Equilibrium ---")

    # 1. Define Product
    widget = Product(name="Widget")

    # 2. Create Producers
    # Producer(name, product, base_cost_per_unit, cost_slope, max_capacity)
    producers = [
        Producer(name="AlphaCorp", product=widget, base_cost_per_unit=2.0, cost_slope=0.05, max_capacity=150),
        Producer(name="BetaWorks", product=widget, base_cost_per_unit=3.0, cost_slope=0.03, max_capacity=200),
        Producer(name="GammaInc", product=widget, base_cost_per_unit=1.5, cost_slope=0.08, max_capacity=100)
    ]

    # 3. Create Consumers
    # Consumer(name, product, max_value_first_unit, value_slope, budget)
    consumers = [
        Consumer(name="Alice", product=widget, max_value_first_unit=15.0, value_slope=0.2, budget=100),
        Consumer(name="Bob", product=widget, max_value_first_unit=12.0, value_slope=0.15, budget=120),
        Consumer(name="Charlie", product=widget, max_value_first_unit=18.0, value_slope=0.25, budget=80),
        Consumer(name="Diana", product=widget, max_value_first_unit=10.0, value_slope=0.1, budget=150)
    ]
    
    # 4. Define Price Range for Simulation
    # Max possible price could be highest max_value_first_unit among consumers
    # Min price could be 0 or lowest base_cost_per_unit among producers
    max_possible_price = max(c.max_value_first_unit for c in consumers)
    min_possible_price = 0 # min(p.base_cost_per_unit for p in producers)
    price_range = np.linspace(min_possible_price, max_possible_price * 1.1, num=50) # Test 50 price points

    # 5. Simulate Market to get Schedules
    demand_schedule, supply_schedule = simulate_market(producers, consumers, widget, price_range)
    
    # 6. Find Equilibrium
    eq_p, eq_q = find_equilibrium(demand_schedule, supply_schedule)

    # 7. Plot
    plot_supply_demand(demand_schedule, supply_schedule, eq_p, eq_q, widget.name)
