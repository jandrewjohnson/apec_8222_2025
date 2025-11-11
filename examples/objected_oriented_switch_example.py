# Define the class "LightSwitch"
class LightSwitch:
    """
    A simple class representing a light switch.
    It can be turned on or off.
    """

    # 1. The Constructor (__init__ method)
    # This method is automatically called when you create a new LightSwitch object.
    # 'self' refers to the instance of the object being created.
    # We can give it initial properties (attributes).
    def __init__(self, location="Default Room"):
        """
        Initializes a new LightSwitch.
        By default, a new light switch is 'off'.
        
        Args:
            location (str): The location of the light switch (e.g., "Living Room", "Kitchen").
        """
        print(f"A new LightSwitch has been installed in the {location}!")
        self.location = location  # Attribute: where the switch is
        self.is_on = False        # Attribute: the state of the light (on/off)

    # 2. Methods (Functions that belong to the class)
    # These methods define what a LightSwitch object can *do*.
    # They always take 'self' as their first argument, allowing them to access
    # and modify the object's attributes.

    def turn_on(self):
        """Turns the light on if it's currently off."""
        if not self.is_on:
            self.is_on = True
            print(f"The light in the {self.location} has been turned ON.")
        else:
            print(f"The light in the {self.location} is already ON.")

    def turn_off(self):
        """Turns the light off if it's currently on."""
        if self.is_on:
            self.is_on = False
            print(f"The light in the {self.location} has been turned OFF.")
        else:
            print(f"The light in the {self.location} is already OFF.")

    def get_status(self):
        """Returns a string describing the current status of the light."""
        if self.is_on:
            return f"The light in the {self.location} is currently ON."
        else:
            return f"The light in the {self.location} is currently OFF."

# --- How to use the LightSwitch class (Demonstration) ---
if __name__ == "__main__":
    print("--- Creating and Using LightSwitch Objects ---")

    # Create an instance (an object) of the LightSwitch class for the living room
    # This calls the __init__ method.
    living_room_switch = LightSwitch(location="Living Room")

    # Create another instance for the kitchen
    kitchen_switch = LightSwitch(location="Kitchen")
    
    print("\n--- Interacting with the Living Room Switch ---")
    # Use the methods of the living_room_switch object
    print(living_room_switch.get_status())  # Check initial status
    living_room_switch.turn_on()            # Turn it on
    print(living_room_switch.get_status())  # Check status again
    living_room_switch.turn_on()            # Try to turn it on again
    living_room_switch.turn_off()           # Turn it off
    print(living_room_switch.get_status())  # Final status

    print("\n--- Interacting with the Kitchen Switch ---")
    # The kitchen_switch is independent of the living_room_switch
    print(kitchen_switch.get_status())      # Check initial status (should be off)
    kitchen_switch.turn_on()
    print(kitchen_switch.get_status())

    # You can access attributes directly (though often methods are preferred for control)
    print(f"\nDirect attribute access: The living room switch is on? {living_room_switch.is_on}")
    print(f"Location of kitchen switch: {kitchen_switch.location}")