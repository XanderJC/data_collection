from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from PIL import Image, ImageTk

import numpy as np
import torch as T
import torchvision as tv
import torchvision.transforms as transforms

import tkinter as tk
import tkinter.ttk as ttk

import matplotlib

matplotlib.use("TkAgg")

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
)

testset = tv.datasets.CIFAR10(
    root="hla/data", train=False, download=True, transform=transform
)

text_preds = np.load("hla/explanations/predictions_saved/text_preds.npy")


class ImageFrame(tk.Frame):
    def __init__(self, container, index):
        super().__init__(container)

        # message = ttk.Label(self, text="This is an image frame")
        # message.grid(column=0, row=1, padx=5, pady=5)

        figure = Figure(figsize=(6, 4), dpi=100)
        figure_canvas = FigureCanvasTkAgg(figure, self)
        NavigationToolbar2Tk(figure_canvas, self)
        axes = figure.add_subplot()

        img, _ = testset[index]

        img = img / 2 + 0.5  # unnormalize

        img = np.transpose(img, (1, 2, 0))
        axes.imshow(img)

        figure_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)


class Explanation(tk.Frame):
    def __init__(self, container, name, image=None):
        super().__init__(container)
        message = ttk.Label(
            self,
            text=name,
            image=image,
            compound="top",
        )
        message.image = image
        # message.grid(column=0, row=0, padx=5, pady=5)
        message.pack()


class ExplanationFrame(tk.Frame):
    def __init__(self, container, index, explainers_list, describe=False):
        super().__init__(container)

        pred = text_preds[index]

        ig = Image.open(f"hla/explanations/ig_saved/ig_test_{index}.png")
        ig = ig.resize((600, 400))
        self.ig = ImageTk.PhotoImage(ig)

        nn = Image.open(f"hla/explanations/nn_saved/nn_test_{index}.png")
        nn = nn.resize((600, 400))
        self.nn = ImageTk.PhotoImage(nn)

        deeplift = Image.open(
            f"hla/explanations/deeplift_saved/deeplift_test_{index}.png"
        )
        deeplift = deeplift.resize((600, 400))
        self.deeplift = ImageTk.PhotoImage(deeplift)

        occlusion = Image.open(
            f"hla/explanations/occlusion_saved/occlusion_test_{index}.png"
        )
        occlusion = occlusion.resize((600, 400))
        self.occlusion = ImageTk.PhotoImage(occlusion)

        simplex = Image.open(f"hla/explanations/simplex_saved/simplex_test_{index}.png")
        simplex = simplex.resize((900, 400))
        self.simplex = ImageTk.PhotoImage(simplex)

        ig_desc = ""
        nn_desc = ""
        dl_desc = ""
        oc_desc = ""
        si_desc = ""

        if describe is True:
            ig_desc = "\nIntegrated Gradients is a method for attributing features \n\
to a model's predictions while satisfying definitions of sensitivity \n\
and implimentation invariance. Higher values indicate greater \n\
importance to the model."
            nn_desc = "\nProvides the example and model prediction of whichever \n\
training set member is closest to the test example in the latent \n\
space of the model."
            dl_desc = "\nDeep Learning Important FeaTures aims to decompose the \n\
prediction into attributions of individual neurons and compare \n\
to a reference attribution in order to determine feature relevance.\n\
Higher values indicate greater importance to the model."
            oc_desc = "\nThis method searches for the minimal mask for the test \n\
example that will result in a different prediction being \n\
outputted by the model."
            si_desc = "\nProvides relevent examples from the training set by \n\
reconstructing the test example's latent representation as a \n\
mixture of the corpus representations."

        self.all_explanations = [
            Explanation(self, "Integrated Gradients" + ig_desc, image=self.ig),
            Explanation(self, "Nearest Neighbour" + nn_desc, image=self.nn),
            Explanation(self, "DeepLIFT" + dl_desc, image=self.deeplift),
            Explanation(self, "Occlusion" + oc_desc, image=self.occlusion),
            Explanation(self, "SimplEx" + si_desc, image=self.simplex),
        ]

        self.selected_explanations = [self.all_explanations[i] for i in explainers_list]
        self.begin = [
            Explanation(self, "Don't click until\nyou make prediction"),
            Explanation(self, f"The ML algorithm predicts:\n{pred}"),
        ]
        self.end = [
            Explanation(self, "No more explanations"),
        ]

        self.explanations = self.begin + self.selected_explanations + self.end

        self.ex_i = 0
        self.explanations[self.ex_i].grid(column=0, row=0, padx=5, pady=5)

        next_button = ttk.Button(
            self, text="Next Explanation", command=self.next_explanation
        )
        next_button.grid(column=0, row=1, padx=5, pady=5)

    def next_explanation(self):
        self.explanations[self.ex_i].grid_forget()
        self.ex_i += 1
        self.explanations[self.ex_i].grid(column=0, row=0, padx=5, pady=5)


class FavouriteFrame(tk.Frame):
    def __init__(self, container, message="Which do you think is most useful?"):
        super().__init__(container)
        self.explainers = (
            " Integrated Gradients",
            " Nearest Neighbour",
            " DeepLIFT",
            " Occlusion",
            " SimplEx",
        )

        self.selected_class = None

        message = ttk.Label(self, text=message)
        message.grid(column=0, row=0, padx=5, pady=5)

        self.selection = self.get_decision_selection()
        self.selection.grid(column=0, row=1, padx=5, pady=5)
        self.selection.bind("<<ListboxSelect>>", self.selected)

    def get_decision_selection(self):

        var = tk.Variable(value=self.explainers)
        return tk.Listbox(
            self,
            listvariable=var,
            selectmode=tk.SINGLE,
            height=5,
            width=20,
            exportselection=False,
        )

    def selected(self, event):
        selected_indices = self.selection.curselection()
        selected_class = self.selection.get(selected_indices)
        self.selected_class = (selected_indices[0], selected_class)


class InteractionFrame(tk.Frame):
    def __init__(self, container):
        super().__init__(container)
        self.classes = (
            " Plane",
            " Car",
            " Bird",
            " Cat",
            " Deer",
            " Dog",
            " Frog",
            " Horse",
            " Ship",
            " Truck",
        )

        self.selected_first = None
        self.selected_second = None

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        message = ttk.Label(self, text="Make your first selection:")
        message.grid(column=0, row=0, padx=5, pady=5)

        self.first_selection = self.get_decision_selection()
        self.first_selection.grid(column=0, row=1, padx=5, pady=5)
        self.first_selection.bind("<<ListboxSelect>>", self.first_selected)

        message = ttk.Label(
            self,
            text="Click through explanations\n\nUntil you make your second selection:",
        )
        message.grid(column=1, row=0, padx=5, pady=5)

        self.second_selection = self.get_decision_selection()
        self.second_selection.grid(column=1, row=1, padx=5, pady=5)
        self.second_selection.bind("<<ListboxSelect>>", self.second_selected)

    def get_decision_selection(self):

        var = tk.Variable(value=self.classes)
        return tk.Listbox(
            self,
            listvariable=var,
            selectmode=tk.SINGLE,
            height=10,
            width=10,
            exportselection=False,
        )

    def first_selected(self, event):
        selected_indices = self.first_selection.curselection()
        selected_class = self.first_selection.get(selected_indices)
        self.selected_first = (selected_indices[0], selected_class)

    def second_selected(self, event):
        selected_indices = self.second_selection.curselection()
        selected_class = self.second_selection.get(selected_indices)
        self.selected_second = (selected_indices[0], selected_class)


class DecisionSelection(tk.Listbox):
    def __init__(self, container):
        super().__init__(container)


class ControlFrame(tk.Frame):
    def __init__(self, container, control_message=None):
        super().__init__(container)
        message = ttk.Label(self, text=control_message)
        message.grid(column=0, row=0, padx=5, pady=5)
        exit_button = ttk.Button(self, text="Next", command=self.master.quit)
        # exit_button.pack(ipadx=5, ipady=5, expand=True)
        exit_button.grid(column=0, row=1, padx=5, pady=5)


class MainDisplay(tk.Tk):
    def __init__(
        self,
        index,
        explainers_list=[0, 1, 2, 3, 4],
        control_message=None,
        describe=False,
    ):
        super().__init__()

        assert index in range(1000), "Index must be between 0 and 999"

        self.index = index
        self.explainers_list = explainers_list
        self.title("Ardent App")
        self.geometry("1500x900")
        # self.resizable(0, 0)  # Disable resizing

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        self.image = ImageFrame(self, index)
        self.image.grid(column=0, row=0, padx=5, pady=5)

        self.explanation = ExplanationFrame(self, index, explainers_list, describe)
        self.explanation.grid(column=1, row=0, padx=5, pady=5)

        self.interaction = InteractionFrame(self)
        self.interaction.grid(column=0, row=1, padx=5, pady=5)

        self.control = ControlFrame(self, control_message)
        self.control.grid(column=1, row=1, padx=5, pady=5)

    def get_results(self):

        max_explainers = len(self.explainers_list)
        first = (
            self.interaction.selected_first[0]
            if self.interaction.selected_first
            else -1
        )
        second = (
            self.interaction.selected_second[0]
            if self.interaction.selected_second
            else -1
        )

        return (
            first,
            second,
            max(0, min(self.explanation.ex_i - 1, max_explainers)),
            self.index,
            self.explainers_list,
        )


class CollectInfo(tk.Tk):
    def __init__(self, selection_message=None, control_message=None):
        super().__init__()

        self.title("Ardent App")
        self.geometry("1500x900")
        self.resizable(0, 0)

        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)

        self.interaction = FavouriteFrame(self, selection_message)
        self.interaction.grid(column=0, row=1, padx=5, pady=5)

        self.control = ControlFrame(self, control_message)
        self.control.grid(column=1, row=1, padx=5, pady=5)
