import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize_scalar
from scipy.integrate import odeint
import pandas as pd
from collections import defaultdict
import random
from dataclasses import dataclass
from typing import List, Tuple
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Common parameters for all models
ALPHA = 0.5  # Capital share in production
BETA = 0.7   # Consumption share in utility
RHO = 0.5    # CES parameter

class GeneralEquilibriumModel:
    """Solves for general equilibrium prices and quantities"""
    
    def __init__(self, labor_endowment=100, capital_endowment=50):
        self.L_bar = labor_endowment
        self.K_bar = capital_endowment
        
    def utility(self, C, L):
        """CES utility function"""
        leisure = self.L_bar - L
        if leisure <= 0:
            return -1e10  # Penalty for no leisure
        return (BETA * C**RHO + (1-BETA) * leisure**RHO)**(1/RHO)
    
    def production(self, L, K):
        """Cobb-Douglas production function"""
        return L**ALPHA * K**(1-ALPHA)
    
    def solve_equilibrium(self):
        """Solve for general equilibrium"""
        def equations(vars):
            w, r, P = vars
            
            # Firm's FOCs
            L_demand = (ALPHA * P / w)**(1/(1-ALPHA)) * self.K_bar
            K_demand = self.K_bar  # All capital used
            
            # Consumer's optimal choices
            # From utility maximization
            income = w * self.L_bar + r * self.K_bar
            C = BETA * income / P
            L_supply = self.L_bar - (1-BETA) * income / w
            
            # Market clearing conditions
            labor_market = L_demand - L_supply
            goods_market = C - self.production(L_demand, K_demand)
            
            # Numeraire: P = 1
            price_norm = P - 1
            
            return [labor_market, goods_market, price_norm]
        
        # Initial guess
        w0, r0, P0 = 1.0, 1.0, 1.0
        
        # Solve
        w_eq, r_eq, P_eq = fsolve(equations, [w0, r0, P0])
        
        # Calculate equilibrium quantities
        income = w_eq * self.L_bar + r_eq * self.K_bar
        C_eq = BETA * income / P_eq
        L_eq = self.L_bar - (1-BETA) * income / w_eq
        Y_eq = self.production(L_eq, self.K_bar)
        
        return {
            'wage': w_eq,
            'rental_rate': r_eq,
            'price': P_eq,
            'consumption': C_eq,
            'labor': L_eq,
            'output': Y_eq,
            'utility': self.utility(C_eq, L_eq)
        }
    
    def get_supply_demand_curves(self, quantity_range):
        """Generate supply and demand curves using partial equilibrium logic"""
        # For visualization purposes, we'll create curves that intersect at equilibrium
        # This uses partial equilibrium logic within the GE model
        
        eq = self.solve_equilibrium()
        eq_Q = eq['output']
        eq_P = eq['price']
        
        # Create supply curve: P = MC (increasing)
        # Assume MC increases with quantity
        supply_elasticity = 0.5
        supply_prices = eq_P * (quantity_range / eq_Q) ** (1/supply_elasticity)
        
        # Create demand curve: P decreases with quantity  
        # Use constant elasticity demand
        demand_elasticity = -0.8
        demand_prices = eq_P * (quantity_range / eq_Q) ** (1/demand_elasticity)
        
        return quantity_range, supply_prices, demand_prices


class SystemsDynamicsModel:
    """Systems dynamics model with stocks and flows"""
    
    def __init__(self, initial_price=1.0, initial_inventory=10):
        self.P0 = initial_price
        self.I0 = initial_inventory
        self.adjustment_speed = 0.1
        self.production_delay = 2.0
        
    def demand_function(self, P, income=100):
        """Demand decreases with price"""
        return income / (P + 1)
    
    def supply_response(self, P, capacity=50):
        """Supply increases with price (with delay)"""
        return capacity * (P / (P + 1))
    
    def dynamics(self, state, t):
        """Define system dynamics"""
        P, I, S = state  # Price, Inventory, Supply rate
        
        # Current demand
        D = self.demand_function(P)
        
        # Inventory change
        dI_dt = S - D
        
        # Price adjustment based on inventory
        target_inventory = 20
        price_pressure = -self.adjustment_speed * (I - target_inventory)
        dP_dt = price_pressure
        
        # Supply adjustment (with delay)
        target_supply = self.supply_response(P)
        dS_dt = (target_supply - S) / self.production_delay
        
        return [dP_dt, dI_dt, dS_dt]
    
    def simulate(self, time_points):
        """Run simulation"""
        initial_state = [self.P0, self.I0, 25]  # Price, Inventory, Supply
        
        solution = odeint(self.dynamics, initial_state, time_points)
        
        results = pd.DataFrame({
            'time': time_points,
            'price': solution[:, 0],
            'inventory': solution[:, 1],
            'supply_rate': solution[:, 2]
        })
        
        # Calculate demand for each time point
        results['demand_rate'] = results['price'].apply(self.demand_function)
        
        return results
    
    def get_supply_demand_at_time(self, t, price_range):
        """Get supply and demand curves at specific time"""
        # Run simulation up to time t
        time_points = np.linspace(0, t, 100)
        results = self.simulate(time_points)
        final_state = results.iloc[-1]
        
        supply = []
        demand = []
        
        for P in price_range:
            demand.append(self.demand_function(P))
            supply.append(self.supply_response(P))
        
        return np.array(supply), np.array(demand), final_state['price']


@dataclass
class Order:
    """Order in the order book"""
    agent_id: int
    quantity: float
    price: float
    is_buy: bool
    
class Producer:
    """Producer agent with profit maximization"""
    
    def __init__(self, agent_id, capital_endowment=10):
        self.id = agent_id
        self.capital = capital_endowment
        self.marginal_cost = 0.5 + 0.1 * agent_id  # Heterogeneous costs
        
    def get_supply_schedule(self, price_range):
        """Generate supply schedule based on profit maximization"""
        orders = []
        for p in price_range:
            if p > self.marginal_cost:
                # Simplified: quantity increases with price above MC
                quantity = self.capital * (p - self.marginal_cost)
                orders.append(Order(self.id, quantity, p, is_buy=False))
        return orders

class Consumer:
    """Consumer agent with utility maximization"""
    
    def __init__(self, agent_id, income=20):
        self.id = agent_id
        self.income = income
        self.preference = 0.8 + 0.05 * agent_id  # Heterogeneous preferences
        
    def get_demand_schedule(self, price_range):
        """Generate demand schedule based on utility maximization"""
        orders = []
        for p in price_range:
            # Simplified: quantity decreases with price
            max_quantity = self.income / p
            desired_quantity = max_quantity * self.preference
            if desired_quantity > 0:
                orders.append(Order(self.id, desired_quantity, p, is_buy=True))
        return orders

class Market:
    """Order book market"""
    
    def __init__(self):
        self.buy_orders = []
        self.sell_orders = []
        self.trades = []
        
    def add_orders(self, orders):
        """Add orders to the book"""
        for order in orders:
            if order.is_buy:
                self.buy_orders.append(order)
            else:
                self.sell_orders.append(order)
    
    def clear_market(self):
        """Match orders using continuous double auction"""
        # Sort orders
        self.buy_orders.sort(key=lambda x: -x.price)  # Highest price first
        self.sell_orders.sort(key=lambda x: x.price)   # Lowest price first
        
        trades = []
        
        while self.buy_orders and self.sell_orders:
            best_buy = self.buy_orders[0]
            best_sell = self.sell_orders[0]
            
            if best_buy.price >= best_sell.price:
                # Trade occurs
                trade_price = (best_buy.price + best_sell.price) / 2
                trade_quantity = min(best_buy.quantity, best_sell.quantity)
                
                trades.append({
                    'price': trade_price,
                    'quantity': trade_quantity,
                    'buyer': best_buy.agent_id,
                    'seller': best_sell.agent_id
                })
                
                # Update quantities
                best_buy.quantity -= trade_quantity
                best_sell.quantity -= trade_quantity
                
                # Remove filled orders
                if best_buy.quantity <= 0:
                    self.buy_orders.pop(0)
                if best_sell.quantity <= 0:
                    self.sell_orders.pop(0)
            else:
                break
        
        self.trades.extend(trades)
        return trades
    
    def get_order_book_curves(self):
        """Extract supply and demand curves from order book"""
        # Aggregate buy orders (demand)
        demand_curve = defaultdict(float)
        for order in self.buy_orders:
            demand_curve[order.price] += order.quantity
        
        # Aggregate sell orders (supply)
        supply_curve = defaultdict(float)
        for order in self.sell_orders:
            supply_curve[order.price] += order.quantity
        
        # Convert to cumulative curves
        if demand_curve:
            prices_d = sorted(demand_curve.keys(), reverse=True)
            quantities_d = []
            cumsum = 0
            for p in prices_d:
                cumsum += demand_curve[p]
                quantities_d.append(cumsum)
        else:
            prices_d, quantities_d = [], []
            
        if supply_curve:
            prices_s = sorted(supply_curve.keys())
            quantities_s = []
            cumsum = 0
            for p in prices_s:
                cumsum += supply_curve[p]
                quantities_s.append(cumsum)
        else:
            prices_s, quantities_s = [], []
        
        return (prices_d, quantities_d), (prices_s, quantities_s)

class ABMModel:
    """Agent-based model with order book market"""
    
    def __init__(self, n_producers=10, n_consumers=20):
        self.producers = [Producer(i, capital_endowment=5+random.random()*10) 
                         for i in range(n_producers)]
        self.consumers = [Consumer(i, income=15+random.random()*10) 
                         for i in range(n_consumers)]
        self.market = Market()
        
    def simulate_one_period(self, price_range):
        """Run one period of the market"""
        # Clear previous orders
        self.market = Market()
        
        # Producers post sell orders
        for producer in self.producers:
            orders = producer.get_supply_schedule(price_range)
            self.market.add_orders(orders)
        
        # Consumers post buy orders
        for consumer in self.consumers:
            orders = consumer.get_demand_schedule(price_range)
            self.market.add_orders(orders)
        
        # Clear market
        trades = self.market.clear_market()
        
        # Get order book curves
        demand_data, supply_data = self.market.get_order_book_curves()
        
        # Calculate average trade price
        if trades:
            avg_price = np.average([t['price'] for t in trades], 
                                 weights=[t['quantity'] for t in trades])
            total_quantity = sum(t['quantity'] for t in trades)
        else:
            avg_price = None
            total_quantity = 0
            
        return {
            'trades': trades,
            'demand_curve': demand_data,
            'supply_curve': supply_data,
            'avg_price': avg_price,
            'total_quantity': total_quantity
        }


def create_comparison_plots():
    """Create plots comparing all three approaches"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Comparison of Economic Modeling Approaches', fontsize=16)
    
    # Common price range for supply/demand curves
    price_range = np.linspace(0.1, 3.0, 50)
    
    # 1. General Equilibrium
    ge_model = GeneralEquilibriumModel()
    ge_results = ge_model.solve_equilibrium()
    
    # Generate quantity range for supply/demand curves
    quantity_range = np.linspace(1, 100, 50)
    quantities, supply_prices, demand_prices = ge_model.get_supply_demand_curves(quantity_range)
    
    ax = axes[0, 0]
    ax.plot(quantities, supply_prices, 'b-', label='Supply', linewidth=2)
    ax.plot(quantities, demand_prices, 'r-', label='Demand', linewidth=2)
    ax.axhline(y=ge_results['price'], color='g', linestyle='--', 
               label=f"Eq. Price = {ge_results['price']:.2f}")
    ax.axvline(x=ge_results['output'], color='g', linestyle='--',
               label=f"Eq. Quantity = {ge_results['output']:.1f}")
    ax.set_xlabel('Quantity')
    ax.set_ylabel('Price')
    ax.set_title('General Equilibrium')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 3)  # Set reasonable y-axis limits
    
    # GE details
    ax = axes[1, 0]
    details = [
        f"Wage: {ge_results['wage']:.3f}",
        f"Rental Rate: {ge_results['rental_rate']:.3f}",
        f"Labor: {ge_results['labor']:.1f}",
        f"Utility: {ge_results['utility']:.2f}"
    ]
    ax.text(0.1, 0.5, '\n'.join(details), transform=ax.transAxes,
            fontsize=12, verticalalignment='center')
    ax.set_title('GE Model Details')
    ax.axis('off')
    
    # 2. Systems Dynamics
    sd_model = SystemsDynamicsModel()
    time_points = np.linspace(0, 50, 200)
    sd_results = sd_model.simulate(time_points)
    
    ax = axes[0, 1]
    supply_sd, demand_sd, current_price = sd_model.get_supply_demand_at_time(20, price_range)
    ax.plot(supply_sd, price_range, 'b-', label='Supply', linewidth=2)
    ax.plot(demand_sd, price_range, 'r-', label='Demand', linewidth=2)
    ax.axhline(y=current_price, color='g', linestyle='--',
               label=f"Current Price = {current_price:.2f}")
    ax.set_xlabel('Quantity')
    ax.set_ylabel('Price')
    ax.set_title('Systems Dynamics (t=20)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # SD time series
    ax = axes[1, 1]
    ax.plot(sd_results['time'], sd_results['price'], 'g-', label='Price')
    ax2 = ax.twinx()
    ax2.plot(sd_results['time'], sd_results['inventory'], 'b--', label='Inventory')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price', color='g')
    ax2.set_ylabel('Inventory', color='b')
    ax.set_title('SD: Price and Inventory Dynamics')
    ax.grid(True, alpha=0.3)
    
    # 3. Agent-Based Model
    abm_model = ABMModel(n_producers=15, n_consumers=25)
    abm_results = abm_model.simulate_one_period(price_range)
    
    ax = axes[0, 2]
    
    # Plot order book
    if abm_results['demand_curve'][0]:
        ax.step(abm_results['demand_curve'][1], abm_results['demand_curve'][0], 
                'r-', label='Demand (Buy Orders)', where='post', linewidth=2)
    if abm_results['supply_curve'][0]:
        ax.step(abm_results['supply_curve'][1], abm_results['supply_curve'][0], 
                'b-', label='Supply (Sell Orders)', where='post', linewidth=2)
    
    if abm_results['avg_price']:
        ax.axhline(y=abm_results['avg_price'], color='g', linestyle='--',
                   label=f"Avg Trade Price = {abm_results['avg_price']:.2f}")
    
    ax.set_xlabel('Quantity')
    ax.set_ylabel('Price')
    ax.set_title('Agent-Based Model (Order Book)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 100)
    
    # ABM trade distribution
    ax = axes[1, 2]
    if abm_results['trades']:
        trade_prices = [t['price'] for t in abm_results['trades']]
        trade_quantities = [t['quantity'] for t in abm_results['trades']]
        
        ax.scatter(trade_quantities, trade_prices, alpha=0.6, s=50)
        ax.set_xlabel('Trade Quantity')
        ax.set_ylabel('Trade Price')
        ax.set_title(f"ABM: Individual Trades (n={len(abm_results['trades'])})")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No trades occurred', transform=ax.transAxes,
                horizontalalignment='center', verticalalignment='center')
        ax.set_title('ABM: Individual Trades')
    
    plt.tight_layout()
    return fig, ge_results, sd_results, abm_results


def create_dynamics_comparison():
    """Compare dynamic behavior across models"""
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Dynamic Behavior Comparison', fontsize=16)
    
    # 1. GE: Comparative statics
    ax = axes[0]
    shocks = np.linspace(0.5, 1.5, 10)
    ge_prices = []
    ge_quantities = []
    
    for shock in shocks:
        ge_model = GeneralEquilibriumModel(labor_endowment=100*shock)
        results = ge_model.solve_equilibrium()
        ge_prices.append(results['price'])
        ge_quantities.append(results['output'])
    
    ax.plot(shocks, ge_prices, 'o-', label='Price')
    ax.set_xlabel('Labor Endowment Shock')
    ax.set_ylabel('Equilibrium Price')
    ax.set_title('GE: Comparative Statics')
    ax.grid(True, alpha=0.3)
    
    # 2. SD: Shock response
    ax = axes[1]
    sd_model = SystemsDynamicsModel(initial_price=1.5)
    time_points = np.linspace(0, 50, 200)
    results = sd_model.simulate(time_points)
    
    ax.plot(results['time'], results['price'], label='Price')
    ax.axhline(y=1.0, color='r', linestyle='--', alpha=0.5, label='Long-run equilibrium')
    ax.set_xlabel('Time')
    ax.set_ylabel('Price')
    ax.set_title('SD: Adjustment to Equilibrium')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. ABM: Price discovery over time
    ax = axes[2]
    abm_model = ABMModel()
    prices_over_time = []
    quantities_over_time = []
    
    price_range = np.linspace(0.1, 3.0, 50)
    
    for period in range(20):
        results = abm_model.simulate_one_period(price_range)
        if results['avg_price']:
            prices_over_time.append(results['avg_price'])
            quantities_over_time.append(results['total_quantity'])
    
    if prices_over_time:
        ax.plot(range(len(prices_over_time)), prices_over_time, 'o-')
        ax.set_xlabel('Trading Period')
        ax.set_ylabel('Average Trade Price')
        ax.set_title('ABM: Price Discovery')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Create comparison plots
    print("Creating model comparison plots...")
    fig1, ge_res, sd_res, abm_res = create_comparison_plots()
    plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
    
    # Create dynamics comparison
    print("Creating dynamics comparison...")
    fig2 = create_dynamics_comparison()
    plt.savefig('dynamics_comparison.png', dpi=300, bbox_inches='tight')
    
    # Print summary statistics
    print("\n=== Model Comparison Summary ===")
    print(f"\nGeneral Equilibrium:")
    print(f"  Equilibrium Price: {ge_res['price']:.3f}")
    print(f"  Equilibrium Quantity: {ge_res['output']:.2f}")
    print(f"  Computation: Instantaneous (solving equations)")
    
    print(f"\nSystems Dynamics:")
    print(f"  Final Price: {sd_res['price'].iloc[-1]:.3f}")
    print(f"  Final Inventory: {sd_res['inventory'].iloc[-1]:.2f}")
    print(f"  Convergence Time: ~20 periods")
    
    print(f"\nAgent-Based Model:")
    if abm_res['avg_price']:
        print(f"  Average Trade Price: {abm_res['avg_price']:.3f}")
        print(f"  Total Quantity Traded: {abm_res['total_quantity']:.2f}")
        print(f"  Number of Trades: {len(abm_res['trades'])}")
        print(f"  Price Dispersion: {np.std([t['price'] for t in abm_res['trades']]):.3f}")
    else:
        print("  No trades occurred")
    
    plt.show()