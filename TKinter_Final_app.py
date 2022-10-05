import tkinter
from tkinter import *
from tkinter import ttk
from tkinter import filedialog, Radiobutton
from PIL import ImageTk, Image
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tensorflow
from tensorflow import keras
from keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import load_img, img_to_array


root = Tk()
root.title("Vision par Ordinateur et Classification d'images")
root.geometry('1024x1440')
my_notebook = ttk.Notebook(root)
# ttk.Style().configure("TNotebook", background=blue, foreground=white)


accueil_tab = Frame(my_notebook, bg="#0c77bd")
ia_tab = Frame(my_notebook, bg="#0c77bd")
usecase_tab = Frame(my_notebook, bg="#0c77bd")
data_tab = Frame(my_notebook, bg="#0c77bd")
quiz_tab = Frame(my_notebook, bg="#0c77bd")

my_notebook.add(accueil_tab, text='Accueil')
my_notebook.add(ia_tab, text="Un programme d'intelligence artificielle")
my_notebook.add(usecase_tab, text="Le cas d'usage")
my_notebook.add(data_tab, text="Les données")
my_notebook.add(quiz_tab, text="Quiz")

my_notebook.pack(expand=1, fill="both")


##### CRÉATION DES FONCTIONS POUR LES BOUTONS #############
def next_page():
    my_notebook.select(ia_tab)

def next_page2():
    my_notebook.select(usecase_tab)

def next_page3():
    my_notebook.select(data_tab)

def next_page4():
    my_notebook.select(quiz_tab)

def center_crop(image_path, size):
    img = Image.open(image_path)
    img = img.resize((size + 1, size + 1))
    x_center = img.width / 2
    y_center = img.height / 2
    size = size / 2
    cr = img.crop((x_center - size, y_center - size, x_center + size, y_center + size))

    return cr


def openchosenfile():
    global my_chosen_image
    global img_path
    # my_chosen_image.grid_forget()
    # This will not actually open the file but get the path of the file
    root.filename = filedialog.askopenfilename(initialdir="/home/siplon/Bureau/Real_Cat_VS_Dog_Image/test_set", title="Choisir un fichier",
                                               filetypes=(("jpg files", "*.jpg"), ("all files", "*.*")))
    # my_label = Label(root, text= root.filename).pack()
    img_path = root.filename
    # Now we will have to open the file with the filename
    # my_image_path = ImageTk.PhotoImage(Image.open(root.filename))
    croped_img = center_crop(root.filename, 300)
    my_chosen_image = ImageTk.PhotoImage(croped_img)
    my_chosen_image2 = Label(slider_frame, image=my_chosen_image)
    my_chosen_image2.grid(row=0, column=0)


def predict_image():
    global proba_chat
    global proba_chien
    classifier = load_model("model_4.h5")

    resize_img = load_img(img_path, target_size=(224, 224))
    resize_arr = img_to_array(resize_img)
    resize_scale = resize_arr/255.0
    resize_modif = np.expand_dims(resize_scale, axis=0)
    proba = classifier.predict(resize_modif)

    proba_chat = 0
    proba_chien = 0
    if proba[0][0] <= 0.5:
        proba_chat = 1-proba[0][0]
    else:
        proba_chien = proba[0][0]

    ##################### graph ############################
    # figsize in inch dpi=dots per inch
    fig = Figure(figsize=(3, 3), tight_layout=True)  # dpi=100)

    x = [proba_chat, proba_chien]
    y = ['Chat', 'Chien']

    plot1 = fig.add_subplot(111)
    plot1.barh(y, x)
    # setting label of y-axis
    # plot1.ylabel("Catégorie")

    # plot1.axes.get_xaxis().set_visible(False)
    plot1.spines['top'].set_visible(False)
    plot1.spines['right'].set_visible(False)
    # plot1.spines['bottom'].set_visible(False)
    plot1.spines['left'].set_visible(False)

    # setting label of x-axis
    # plot1.xlabel("Probabilité de la catégorie")
    if proba[0][0] <= 0.5:
        plot1.set_title("Catégorie prédite : CHAT ", fontdict={'color': 'red', 'fontweight': 'bold'})
    else:
        plot1.set_title("Catégorie prédite : CHIEN", fontdict={'color': 'red', 'fontweight': 'bold'})
    # plt.show()

    # getting values against each value of y
    canvas = FigureCanvasTkAgg(fig, master=prediction_frame)
    canvas.draw()

    canvas.get_tk_widget().grid(row=2, column=1)

def get_score():
    if v1.get() == 101 and v2.get() == 201 and v3.get() == 300 and v4.get() == 402:
        score = Label(quiz_tab, text='Tu as fait un sans fautes !',
                        font=("Times New Roman", 30, "bold"),
                        background="#0c77bd", foreground="red", borderwidth=4, relief='solid')
        score.grid(row=8, column=2)
    else:
        score = Label(quiz_tab, text='Il y a encore des erreurs !',
                            font=("Times New Roman", 30, "bold"),
                            background="#0c77bd", foreground="red", borderwidth=4,relief='solid')
        score.grid(row=8, column=2)


##### CRÉATION DES ONGLETS ET BOUTON DE L'ACCUEIL #########
title = Label(accueil_tab, text="Vision par ordinateur et classification d'images",
              font=("Times New Roman", 30, "bold"),
              background="#0c77bd", foreground="white")

subtitle = Label(accueil_tab,
                 text="La classification d'images de chats et de chiens comme outil de médiation pédagogique.",
                 font=("Times New Roman", 28, "bold", "italic"), background="#0c77bd", foreground="white")

first_p = Label(accueil_tab,
                text="Cette application a été conçue afin de faciliter l'utilisation de la classification d'images comme\n"
                     "outil pédagogique dans les classes et les environnements où il est difficile d'accéder à internet.\n"
                     "En effet, la majorité des programmes d'intelligence artificielle requête un serveur cloud pour\n"
                     "obtenir une prédiction.",
                font=("Times New Roman", 25), background="#0c77bd", foreground="white")

second_p = Label(accueil_tab,
                 text="Cet outil permet d'apporter un éclairage sur des éléments clés de la conception d'un programme\n"
                      "d'intelligence artificielle. Il a pour vocation de montrer l'importance de la définition du cas\n"
                      "d'usage tout en insistant sur la nécessité de récolter des données de qualités et en quantité.\n"
                      "Il permet aussi d'aborder le choix crucial du matériel et l'étape clé de sélection du modèle le\n"
                      "plus adapté.",
                 font=("Times New Roman", 25), background="#0c77bd", foreground="white")

start_btn = Button(accueil_tab, text="Démarrer", font=("Times New Roman", 20, "bold", 'italic'),
                   foreground='black', background="white", command=next_page)

logo_MIA = ImageTk.PhotoImage(Image.open("logo_MIA128.png"))
my_MIA_img = Label(accueil_tab, image=logo_MIA)

logo_AEC = ImageTk.PhotoImage(Image.open("AEC120.png"))
my_AEC_img = Label(accueil_tab, image=logo_AEC)

my_MIA_img.grid(row=0, column=0, padx=10, pady=20)
my_AEC_img.grid(row=0, column=3, padx=10, pady=20)
title.grid(column=2, row=0, padx=150, pady=30)
subtitle.grid(column=2, row=1, )
first_p.grid(column=2, row=2, pady=80)
second_p.grid(column=2, row=3)
start_btn.grid(column=2, row=4, pady=70)

######## ONGLET IA ###############################
ia_titre = Label(ia_tab, text="Programmes d'intelligences artificielles : quelques éléments clés",
                 font=("Times New Roman", 35, "bold"), background="#0c77bd", foreground="white")

ia_first_p = Label(ia_tab,
                   text="La conception d'un programme d'intelligence artificielle est un processus itératif et cyclique\n"
                        "qui permet de garantir la sureté et la fiabilité de l'application tout en permettant des\n"
                        "améliorations fonctionnelles incrémentales. Cette démarche permet de proposer un modèle qui est\n"
                        "à jour tout en contrôlant les coûts relatifs au matériel, à l'entrainement et à la prédiction.",
                   font=("Times New Roman", 25), background="#0c77bd", foreground="white")

ia_second_p = Label(ia_tab,
                    text="Après une idée d'un programme d'IA qui répondrait à un cas d'usage, le développeur en IA\n"
                         "code le programme puis se doit de le tester afin de garantir à tout instant l'accès à un\n"
                         "programme sûr et fonctionnel avec un modèle réalisant des prédictions justes et sans biais.\n"
                         "Cette étape de test permet aussi d'implémenter des améliorations fonctionnelles et esthétiques\n"
                         "mais peut aussi permettre d'affiner le cas d'usage grâce aux retours d'expériences. Ces phases\n"
                         "de tests sont le moyen privilégié pour maintenir le lien et répondre aux besoins de la réalité\n"
                         "du terrain.",
                    font=("Times New Roman", 25), background="#0c77bd", foreground="white")

ia_third_p = Label(ia_tab, text="Selon vous quels éléments peuvent permettre d'exécuter plus rapidement le cycle\n"
                                "                    Idée - Code - Expérimentation ?",
                   font=("Times New Roman", 25, 'bold'), background="#0c77bd", foreground="red")

first_image = ImageTk.PhotoImage(Image.open("image2graph.png"))
my_first_img = Label(ia_tab, image=first_image)

usecase_btn = Button(ia_tab, text="Suite", foreground='black', background="white",
                     font=("Times New Roman", 20, "bold", "italic"), command=next_page2)

my_first_img.grid(column=0, row=2, padx=20, pady=50)
ia_titre.grid(column=1, row=0, padx=25, pady=20)
ia_first_p.grid(column=1, row=1, pady=25)
ia_second_p.grid(column=1, row=2)
ia_third_p.grid(column=1, row=3)

usecase_btn.grid(column=1, row=4, pady=40, padx=50)

################ ONGLET CAS D'USAGE ###############################
usecase_title = Label(usecase_tab, text="Le cas d'usage: un choix primordial", font=("Times New Roman", 35, "bold"),
                      background="#0c77bd", foreground="white")
usecase_first_p = Label(usecase_tab,
                        text="La vision par ordinateur permet de répondre à de nombreux cas d'usages répandus à travers\n"
                             "tous les domaines d'activités de notre société. Dans la santé, les programme d'IA sont notamment\n"
                             "capables de détecter des cancers ou des mélanomes. Dans l'industrie, les programmes permettant\n"
                             "de faire des contrôles (qualités ou maintenance) sont très utilisés. Dans le domaine de l'agriculture\n"
                             "de précision, son couplage à des drones permet d'apporter des solutions écologiques à des enjeux\n"
                             "de réchauffement climatique.\n\n"
                             "\nIl est donc absolument indispensable de bien définir le cas d'usage avant de se lancer dans la\n"
                              "conception du programme d'intelligence artificielle. En effet, même si on souhaite réaliser une\n"
                              "même tâche (par exemple, classifier des images) la complexité peut varier grandement en fonction\n"
                              "du type de classification et de l'élément à classifier. Il est plus simple de réaliser un programme\n"
                              "de classification de chats et de chiens que d'espèces plus rares ou plus ressemblantes. De même,\n"
                              "classifier une image en fonction de l'objet principal est différent de classifier une image\n"
                              "en fonction du contexte.",
                        font=("Times New Roman", 22), background="#0c77bd", foreground="white")

usecase_question = Label(usecase_tab,
                         text="Ce programme classifie des images de chats et de chiens. Choisissez d'abord une\n"
                              "image à classifier puis cliquer sur 2. Je lance la prédiction afin d'obtenir\n"
                              "le résultat. Que se passe-t-il si on demande au programme de classifier une image\n"
                              "d'une moto ? Pourquoi ?",
                         font=("Times New Roman", 22, 'bold'), background="#0c77bd", foreground="red")

usecase_frame = LabelFrame(usecase_tab, text="Classification d'images", height=300, width=300)
slider_frame = Frame(usecase_frame, background="grey", height=300, width=300)

prediction_frame = LabelFrame(usecase_frame, background="grey", heigh=300, width=300)

choose_img_btn = Button(usecase_frame, text="1. Je choisis une image.", background="white",
                        foreground="red",
                        font=("Times New Roman", 15, "bold"),
                        command=openchosenfile)

predict_btn = Button(usecase_frame, text="2. Je lance la prédiction.", background="white",
                     foreground="red",
                     font=("Times New Roman", 15, 'bold'),
                     command=predict_image)

data_btn = Button(usecase_tab, text="Suite", foreground='black', background="white",
                  font=("Times New Roman", 20, "bold", "italic"), command=next_page3)

usecase_title.grid(row=0, column=0)
usecase_first_p.grid(row=1, column=0, padx=30, pady=10)
usecase_question.grid(row=2, column=0, padx=30)

usecase_frame.grid(row=1, column=1, padx=80)

data_btn.grid(row=2, column=1, pady=60, padx=160)

slider_frame.grid(row=0, column=1, padx=10, pady=10)
choose_img_btn.grid(row=1, column=1)
prediction_frame.grid(row=2, column=1, padx=10, pady=10)
predict_btn.grid(row=3, column=1)

################## ONGLET LES DONNÉES #####################
data_title = Label(data_tab, text="La récolte de données: de qualité et en quantité",
                   font=("Times New Roman", 35, "bold"),
                   background="#0c77bd", foreground="white")

first_p_data = Label(data_tab,
                     text="Dans les programmes d'intelligence artificielle utilisant de l'apprentissage supervisé\n"
                          "c'est-à-dire un apprentissage basé sur des exemples, c'est au développeur d'alimenter\n"
                          "le programme d'IA avec des données les plus justes afin que le programme ne soit pas\n"
                          "sensible aux biais et avec des données en quantités afin qu'une prédiction juste puisse\n"
                          "être réalisée sur des images que le programme n'aura jamais vu: c'est la généralisation.\n"
                          "Le développement des méthodes d'apprentissage profond a été permis par deux éléments\n"
                          "principaux: La disponibilité des ressources computationnelles (serveur de calcul, machine \n"
                          "puissante, GPU etc.) ainsi que la disponibilité des données (trafic internet, réseaux sociaux,\n"
                          "publications etc.).\n\n"
                          "Par ailleurs, il parait logique de penser que plus il existe de données plus un apprentissage\n"
                          "sera efficace. En réalité, le graphique montre que les méthodes traditionnelles et d'apprentissage\n"
                          "machine (machine learning) arrivent très vite à un plateau de performance lorsque la quantité de\n"
                          "données augmente. Ce sont les méthodes d'apprentissage profond (deep learning) qui arrivent le\n"
                          "mieux à exploiter ces big data afin d'en tirer les meilleures performances.\n\n"
                          "Dans le domaine de l'intelligence artificielle un biais représente un écart entre le résultat\n"
                          "du programme et l'intention du cas d'usage. Ainsi, un biais peut survenir lorsque les données\n"
                          "ne sont pas équilibrées où ne sont pas représentatives du cas d'usage. Il convient pour éviter\n"
                          "ces biais de vérifier les données d'entrainement de constituer des équipes pluridisciplinaires\n"
                          "et surtout de réaliser des phases de tests de non régression de l'application et du modèle.",
                     font=("Times New Roman", 22), background="#0c77bd", foreground="white")

question_data = Label(data_tab, text="Selon vous quel graphique correspond au meilleur modèle ?\n"
                                     "Selon vous quel graphique correspond au modèle entrainé avec le plus de données ?",
                      font=("Times New Roman", 22, 'bold'), background="#0c77bd", foreground="red")

material_btn = Button(data_tab, text="Passer au Quiz", foreground='black', background="white",
                      font=("Times New Roman", 20, "bold", "italic"), command=next_page4)

data_image = ImageTk.PhotoImage(Image.open("data_impact50.png"))
my_data_img = Label(data_tab, image=data_image)

data_title.grid(row=0, column=0, pady=30)
first_p_data.grid(row=1, column=0, padx=30)
question_data.grid(row=2, column=0)
my_data_img.grid(row=1, column=1)
material_btn.grid(row=2, column=1, pady=60)


##### ONGLET QUIZ #######
quiz_title = Label(quiz_tab, text=" QUIZ\n"
                                  "Répondez aux 4 questions en essayant de faire un sans faute !",
                     font=("Times New Roman", 35, "bold"),
                     background="#0c77bd", foreground="white")

v1 = tkinter.IntVar()
q1 = Label(quiz_tab, text="Q1. La vision par ordinateur représente : ",
           font=("Times New Roman", 16, "bold"),
           background="#0c77bd", foreground='white', anchor='w')

q1_ans1 = Radiobutton(quiz_tab,
                      text="Un ordinateur équipé d’une webcam.",
                      variable = v1,
                      value=100,
                      anchor='w',
                      background="#0c77bd", foreground='white',
                      borderwidth=0,highlightthickness=0,
                      pady=15,selectcolor='red')
q1_ans2 = Radiobutton(quiz_tab,
                      text="Une des fonctions d’intelligence artificielle",
                      variable=v1,
                      value=101,
                      anchor='w',
                      background="#0c77bd", foreground='white',
                      borderwidth=0,highlightthickness=0,
                      pady=15,selectcolor='red')
q1_ans3 = Radiobutton(quiz_tab,
                      text="La capacité d’un robot à comprendre ce qu’il voit",
                      variable=v1,
                      value=102,
                      anchor='w',
                      background="#0c77bd", foreground='white',
                      borderwidth=0,highlightthickness=0,
                      pady=20,selectcolor='red')



v2 = tkinter.IntVar()
q2 = Label(quiz_tab, text="Q2. Si je montre une image d’une moto à un programme classifiant des images de chats et de chiens :",
          font=("Times New Roman", 16, "bold"),
          background="#0c77bd", foreground='white', anchor='w')

q2_ans1 = Radiobutton(quiz_tab,
                      text="L’algorithme va reconnaître qu’il s’agit d’une moto.",
                      variable = v2,
                      value=200,
                      anchor='w',
                      background="#0c77bd", foreground='white',
                      borderwidth=0,highlightthickness=0,
                      pady=15,selectcolor='red')

q2_ans2 = Radiobutton(quiz_tab,
                      text="L’algorithme va le classifier comme un chat ou un chien.",
                      variable = v2,
                      value=201,
                      anchor='w',
                      background="#0c77bd", foreground='white',
                      borderwidth=0,highlightthickness=0,
                      pady=15,selectcolor='red')

q2_ans3 = Radiobutton(quiz_tab,
                      text="L’algorithme ne donnera pas de réponse.",
                      variable = v2,
                      value=202,
                      anchor='w',
                      background="#0c77bd", foreground='white',
                      borderwidth=0,highlightthickness=0,
                      pady=20,selectcolor='red')


v3 = tkinter.IntVar()
q3 = Label(quiz_tab, text="Q3. Si la précision de la prédiction de mon programme d’intelligence artificielle n’est pas suffisante :",
           font=("Times New Roman", 16, "bold"),
           background="#0c77bd", foreground='white', anchor='w')

q3_ans1 = Radiobutton(quiz_tab,
                      text="Je dois d’abord augmenter le nombre d’images pour l’entraînement.",
                      variable = v3,
                      value=300,
                      anchor='w',
                      background="#0c77bd", foreground='white',
                      borderwidth=0,highlightthickness=0,
                      pady=15,selectcolor='red')

q3_ans2 = Radiobutton(quiz_tab,
                      text="Je dois d’abord diminuer le nombre d’images pour l’entraînement.",
                      variable = v3,
                      value=301,
                      anchor='w',
                      background="#0c77bd", foreground='white',
                      borderwidth=0,highlightthickness=0,
                      pady=15,selectcolor='red')

q3_ans3 = Radiobutton(quiz_tab,
                      text="Je dois d’abord acheter un ordinateur plus puissant.",
                      variable = v3,
                      value=302,
                      anchor='w',
                      background="#0c77bd", foreground='white',
                      borderwidth=0,highlightthickness=0,
                      pady=20,selectcolor='red')


v4 = tkinter.IntVar()
q4 = Label(quiz_tab, text="Q4. Quel élément influence la qualité de l’apprentissage d’un programme d’intelligence artificielle ?",
           font=("Times New Roman", 16, "bold"),
           background="#0c77bd", foreground='white', anchor='w')

q4_ans1 = Radiobutton(quiz_tab,
                      text="Uniquement la qualité des données d’apprentissage.",
                      variable = v4,
                      value=400,
                      anchor='w',
                      background="#0c77bd", foreground='white',
                      borderwidth=0,highlightthickness=0,
                      pady=15,selectcolor='red')

q4_ans2 = Radiobutton(quiz_tab,
                      text="Uniquement la quantité des données d’apprentissage.",
                      variable = v4,
                      value=401,
                      anchor='w',
                      background="#0c77bd", foreground='white',
                      borderwidth=0,highlightthickness=0,
                      pady=15,selectcolor='red')
q4_ans3 = Radiobutton(quiz_tab,
                      text="La qualité et la quantité des données d’apprentissages",
                      variable = v4,
                      value=402,
                      anchor='w',
                      background="#0c77bd", foreground='white',
                      borderwidth=0,highlightthickness=0,
                      pady=20,selectcolor='red')

get_score_btn = Button(quiz_tab, text="Sans fautes ou pas ?",
                          foreground='black', background="white", font=("Times New Roman", 20, "bold", "italic"),
                          command=get_score)

quiz_title.grid(row=0, pady=30, padx=50)

q1.grid(row=1)
q1_ans1.grid(row=2)
q1_ans2.grid(row=3)
q1_ans3.grid(row=4)

q2.grid(row=5)
q2_ans1.grid(row=6)
q2_ans2.grid(row=7)
q2_ans3.grid(row=8)

q3.grid(row=9)
q3_ans1.grid(row=10)
q3_ans2.grid(row=11)
q3_ans3.grid(row=12)

q4.grid(row=13)
q4_ans1.grid(row=14)
q4_ans2.grid(row=15)
q4_ans3.grid(row=16)
get_score_btn.grid(row=18, pady=30)

root.mainloop()


