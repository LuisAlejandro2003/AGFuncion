from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QPushButton , QTextEdit
from algorithm import GeneticAlgorithm
from prettytable import PrettyTable

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()

        self.initUI()

    def initUI(self):
        self.setWindowTitle('Algoritmo Genético')
        self.resize(500, 300)

        layout = QVBoxLayout()
        self.label_subtitulo = QLabel('Datos de la poblacion')
        layout.addWidget(self.label_subtitulo)

        self.label_inicial = QLabel('Valor inicial:')
        self.input_inicial = QLineEdit()

        self.label_final = QLabel('Valor maximo:')
        self.input_final = QLineEdit()

        self.label_generaciones = QLabel('Generaciones:')
        self.input_generaciones = QLineEdit()

        layout.addWidget(self.label_inicial)
        layout.addWidget(self.input_inicial)
        layout.addWidget(self.label_final)
        layout.addWidget(self.input_final)
        layout.addWidget(self.label_generaciones)
        layout.addWidget(self.input_generaciones)

        self.label_subtitulo_mutacion = QLabel('Datos de mutacion')
        self.label_subtitulo_mutacion.setStyleSheet('font-weight: bold')
        layout.addWidget(self.label_subtitulo_mutacion)

        self.label_mutacion_individual = QLabel('Probabilidad de mutacion individual:')
        self.input_mutacion_individual = QLineEdit()
        layout.addWidget(self.label_mutacion_individual)
        layout.addWidget(self.input_mutacion_individual)

        self.label_mutacion_por_gen = QLabel('Probabilidad de mutacion por gen:')
        self.input_mutacion_por_gen = QLineEdit()
        layout.addWidget(self.label_mutacion_por_gen)
        layout.addWidget(self.input_mutacion_por_gen)

        self.label_subtitulo_operacion = QLabel('Datos de la operacion')
        layout.addWidget(self.label_subtitulo_operacion)

     

        self.button_iniciar = QPushButton('Start')
        layout.addWidget(self.button_iniciar)

        # Agregar un QTextEdit para mostrar la información de la población
        self.population_info = QTextEdit()
        self.population_info.setReadOnly(True)
        layout.addWidget(self.population_info)

        self.setLayout(layout)
        self.button_iniciar.clicked.connect(self.start_genetic_algorithm)

    def start_genetic_algorithm(self):
        self.population_info.clear()
        initial_value = float(self.input_inicial.text())
        final_value = float(self.input_final.text())
        generations = int(self.input_generaciones.text())
        individual_mutation = float(self.input_mutacion_individual.text())
        mutation_per_gen = float(self.input_mutacion_por_gen.text())

        algorithm = GeneticAlgorithm(initial_value, final_value, generations, individual_mutation,
                                     mutation_per_gen)

        results = algorithm.run()
        for result in results:
            info_text = f"""
            Generación {result['generation']}: Mejor individuo: {result['best_individual']}
            Error del mejor individuo de la generación {result['generation']}: {result['best_individual_error']}
            """
            self.population_info.append(info_text)