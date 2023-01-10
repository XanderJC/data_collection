from hla import TEST_INDEX
from hla.gui import MainDisplay, CollectInfo
from hla.controllers import RandomController, ArdentController
import numpy as np
import dill

choice = np.random.binomial(1, 0.5)
choice = 0
if choice == 0:
    method = RandomController()
else:
    method = ArdentController()


name = input("Please enter your name:")

message = "This is an example test image. \n\
Please click through the explanations \n\
to have a look at how the system works \n\
and what the possible explanations \n\
look like. Each explanation has a \n\
short description of how it works."

s_message = "Now you have seen how the system works, \n\
please select which explanation you think is most useful."

e_message = "Thanks for completing the task. \n\
Please select which explanation you found most useful, \n\
then please press 'Next' (DO NOT close the window)"


app = MainDisplay(0, list(range(5)), message, describe=True)
app.mainloop()
app.destroy()


info = CollectInfo(selection_message=s_message)
info.mainloop()
initial_pick = info.interaction.selected_class

info.destroy()


indices = TEST_INDEX
total = len(indices)

for i, indx in enumerate(indices):

    explainers = method.select_explainers()

    app = MainDisplay(indx, explainers, f"Example ({i+1}/{total})")
    app.mainloop()

    res = app.get_results()
    app.destroy()

    method.update(*res)

memory = method.memory

info_end = CollectInfo(selection_message=e_message)
info_end.mainloop()
final_pick = info_end.interaction.selected_class
info_end.destroy()


results = {
    "traj": memory,
    "name": name,
    "method": choice,
    "initial_pick": initial_pick,
    "final_pick": final_pick,
}
save_path = f"results_{name}.obj"

with open(save_path, "wb") as f:
    dill.dump(results, f)
