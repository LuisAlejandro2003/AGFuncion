from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QLineEdit, QComboBox, QPushButton , QTextEdit
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

        self.label_optimizacion = QLabel('Tipo de optimizacion:')
        self.input_optimizacion = QComboBox()
        self.input_optimizacion.addItems(['Minimizacion', 'Maximizacion'])

        layout.addWidget(self.label_inicial)
        layout.addWidget(self.input_inicial)
        layout.addWidget(self.label_final)
        layout.addWidget(self.input_final)
        layout.addWidget(self.label_generaciones)
        layout.addWidget(self.input_generaciones)
        layout.addWidget(self.label_optimizacion)
        layout.addWidget(self.input_optimizacion)

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

   
        self.label_limite_inicial_x = QLabel('Limite inicial en X:')
        self.input_limite_inicial_x = QLineEdit()

        self.label_limite_maximo_y = QLabel('Limite maximo en Y:')
        self.input_limite_maximo_y = QLineEdit()

        self.label_resolucion_referencia = QLabel('Resolucion de referencia:')
        self.input_resolucion_referencia = QLineEdit()

        layout.addWidget(self.label_limite_inicial_x)
        layout.addWidget(self.input_limite_inicial_x)
        layout.addWidget(self.label_limite_maximo_y)
        layout.addWidget(self.input_limite_maximo_y)
        layout.addWidget(self.label_resolucion_referencia)
        layout.addWidget(self.input_resolucion_referencia)

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
        optimization = self.input_optimizacion.currentText()
        individual_mutation = float(self.input_mutacion_individual.text())
        mutation_per_gen = float(self.input_mutacion_por_gen.text())
        initial_limit_x = float(self.input_limite_inicial_x.text())
        maximum_limit_y = float(self.input_limite_maximo_y.text())
        reference_resolution = float(self.input_resolucion_referencia.text())

        algorithm = GeneticAlgorithm(initial_value, final_value, generations, optimization, individual_mutation,
                                     mutation_per_gen, initial_limit_x, maximum_limit_y, reference_resolution)

        all_generations_info = algorithm.run()
        for generation_info in all_generations_info:
            info_text = f"""
            Generacion: {generation_info['generation']}
            Poblacion: {generation_info['population']}
            Parejas formadas: {generation_info['pairs_population']}
            Población despues cruce: {generation_info['crossover_population']}
            Población despues mutación: {generation_info['mutation_population']}
            Poblacion despues de la poda: {generation_info['last_population']}
           
            Mejor individuo: {generation_info['best_global_individual']}
            Mejor fitness: {generation_info['best_fitness']}
            Peor fitness: {generation_info['worst_fitness']}
            Fitness promedio: {generation_info['average_fitness']}
            """

            # Agregar la información de la generación al QTextEdit
            self.population_info.append(info_text)

        # Crear una tabla con los encabezados deseados
        
        
        table = PrettyTable(["Cadena de bits", "Índice", "Valor de x", "Valor de aptitud"])

        # Añadir una fila a la tabla para el mejor individuo global
        table.add_row(generation_info['best_global_individual'])
        print("Mejor individuo global:")
        print(table)