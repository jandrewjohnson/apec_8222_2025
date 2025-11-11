# Define the class "LightSwitch"
class LightSwitch:
    def __init__(self, location="Default Room"):
        self.location = location  # Attribute: where the switch is
        self.is_on = False        # Attribute: the state of the light (on/off)
            
    def flip_switch(self):
        if self.is_on:
            self.is_on = False
        else:
            self.is_on = True

    def get_status(self):
        if self.is_on:
            return f"The light in the {self.location} is currently ON."
        else:
            return f"The light in the {self.location} is currently OFF."

    
# Create an instance (an object) of the LightSwitch class for two rooms. It starts turned-off.
living_room_switch = LightSwitch(location="Living Room")
kitchen_switch = LightSwitch(location="Kitchen")

# Do some stuff to the switches
living_room_switch.flip_switch()  # Turn it on
kitchen_switch.flip_switch()  # Turn it on
kitchen_switch.flip_switch()  # Turn it off

# Check the status of the switches
print(living_room_switch.get_status())  
print(kitchen_switch.get_status())  

# Simple exercise:
# Create three switches for three rooms. Flip just the second swithc. Report the status of them all after.

# Less simple exercise:
# Create n switches for n rooms. 
# Flip each switch randomly with 50% probability.
# Report the status of them all after. 
# Do this without having any lines that feel repetitive.
# Hint: Use a for loop, appending the new objects into an existing list that was created before the loop.

import random
switches = []
n = 10
for i in range(n):
    switches.append(LightSwitch(location=f"Room {i}"))
    
for switch in switches:
    random_flip = random.randint(0, 1)
    if random_flip == 1:
        switch.flip_switch()

for switch in switches:
    print(switch.get_status())
    

# print([i.get_status() for i in switches])