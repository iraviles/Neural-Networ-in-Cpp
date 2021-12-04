//Red Neuronal - Fully Connected and Feed Forward

#include<iostream>
#include<vector>
#include<cstdlib>
#include<cassert>
#include<cmath>

using namespace std;

class Neuron;
typedef vector<Neuron> Layer;

struct Connection{
    double weight;
    double deltaWeight;
};


class Neuron{
public:
    Neuron(unsigned numOutputs, unsigned myIndex);    //(Número de salidas de la neurona)
    void setOutputVal(double val){m_outputVal = val;}
    double getOutputVal(void) const {return m_outputVal;}
    void feedForward(const Layer & prevLayer);
    void calcOutputGradients(double targetVal);
    void calcHiddenGradients(const Layer &nextLayer);
    void updateInputWeights(Layer &prevLayer);

private:
    static double eta;
    static double alpha;
    static double transferFunction(double x);
     static double transferFunctionDerivative(double x);
    static double randomWeight(void){return rand() / double(RAND_MAX);} //Valores aleatorios. Requiere cstdlib
    double sumDOW(const Layer &nextLayer)const;
    double m_outputVal;                 //Valor de salida
    vector<Connection> m_outputWeights;     //Valores de los pesos. No se usa una estructura extra dado que las capas son fully connected
    unsigned m_myIndex;
    double m_gradient;
};

double Neuron::eta = 0.15;          //Valor fijo en todo el entrenamiento;
double Neuron::alpha = 0.5;


void Neuron::updateInputWeights(Layer &prevLayer){
    //Los pesos a actualizarse están en el contenedor Connection, en las neuronas de la capa previa
    for(unsigned n = 0; n < prevLayer.size(); ++n){
        Neuron &neuron = prevLayer[n];
        double oldDeltaWeight = neuron.m_outputWeights[m_myIndex].deltaWeight;

        double newDeltaWeight =
                    //Entrada individual, aumentada por el gradiente y la tasa de entrenamiento
                    eta
                    * neuron.getOutputVal()
                    * m_gradient
                    //Tambien se añade momentum = ua fraction of the previous delta weight
                    + alpha
                    * oldDeltaWeight;

        neuron.m_outputWeights[m_myIndex].deltaWeight = newDeltaWeight;
        neuron.m_outputWeights[m_myIndex].weight = newDeltaWeight;
    }
}

double Neuron::sumDOW(const Layer &nextLayer) const{
    double sum = 0.0;

    //Sumar las contribuciones de los errores en los nodos alimentados
    for(unsigned n = 0; n < nextLayer.size() -1; ++n){
        sum += m_outputWeights[n].weight * nextLayer[n].m_gradient;
    }
    return sum;
}

void Neuron::calcHiddenGradients(const Layer &nextLayer){
    double dow = sumDOW(nextLayer);
    m_gradient = dow * Neuron::transferFunctionDerivative(m_outputVal);
}

void Neuron::calcOutputGradients(double targetVal){
    double delta = targetVal - m_outputVal;
    m_gradient = delta * Neuron::transferFunctionDerivative(m_outputVal);
}

double Neuron::transferFunction(double x){
    // Tangente hiperbólica
    return tanh(x);
}

double Neuron::transferFunctionDerivative(double x){
    // d(tanh) = 1 - tanh^2
    return 1.0 - tanh(x) * tanh(x);
}

void Neuron::feedForward(const Layer & prevLayer){
    double sum = 0.0;
    //Suma las salidas de la capa anterior (entradas actuales)
    //Incluir la neurona bias de la capa previa
    for(unsigned n = 0; n < prevLayer.size(); ++n){
        sum += prevLayer[n].getOutputVal() *
               prevLayer[n].m_outputWeights[m_myIndex].weight;
    }

    m_outputVal = Neuron::transferFunction(sum);
}


Neuron::Neuron(unsigned numOutputs, unsigned myIndex){
    for(unsigned c = 0; c< numOutputs; ++c){
        m_outputWeights.push_back(Connection());    //Se agrega un elemento nuevo de tipo conexión en cada iteración
        m_outputWeights.back().weight = randomWeight(); //Se inicializan los pesos con un númeor aleatorio
    }

    myIndex = myIndex;
}

class Net{
public:
    Net(const vector<unsigned> &topology);
    void feedForward(const vector<double> & inputVals);
    void backprop(const vector<double> & targetVals);
    void getResults(vector<double> & resultVals);       //Se usa consta ya que esta función miembro NO modifica los resultados

private:
    vector<Layer> m_layers;        //m_layers[layerNum][neuronNum]
    double m_error;
    double m_recentAverageError;
    double m_recentAverageSmoothingFactor;

};

void Net::getResults(vector<double> & resultVals){
    resultVals.clear();

    for(unsigned n = 0; n < m_layers.back().size() - 1; ++n){
        resultVals.push_back(m_layers.back()[n].getOutputVal());
    }
}


void Net::backprop(const vector<double> & targetVals){
    //Cálculo del error neto (Root Mean Square Error) de las neuronas de salida
    Layer & outputLayer = m_layers.back();
    m_error = 0.0;

    for(unsigned n = 0; n < outputLayer.size() - 1; ++n){
        double delta = targetVals[n] - outputLayer[n].getOutputVal();       //Diferencia entre el valor real y el predicho
        m_error += delta * delta;
    }
    m_error /= outputLayer.size() - 1;                                      //Error cuadrático medio
    m_error = sqrt(m_error);                                                //Root Mean Square

    //Indicador del desempeño de la red
    m_recentAverageError =
            (m_recentAverageError * m_recentAverageSmoothingFactor + m_error)
            / (m_recentAverageSmoothingFactor + 1.0);

    //Cálculo de gradiente de capa de salida
    for(unsigned n = 0; n < outputLayer.size() - 1; ++n){
        outputLayer[n].calcOutputGradients(targetVals[n]);
    }

    //Cálculo de gradiente en capas ocultas
    for(unsigned layerNum = outputLayer.size() - 2; layerNum > 0; --layerNum){
       Layer &hiddenLayer = m_layers[layerNum];                     //Capa actual en la iteración
       Layer &nextLayer   = m_layers[layerNum + 1];

       for(unsigned n = 0; n < hiddenLayer.size(); ++n){
            hiddenLayer[n].calcHiddenGradients(nextLayer);
       }
    }

    //Actualizar pesos desde la capa de salida a la primer capa oculta
    for (unsigned layerNum = m_layers.size() - 1; layerNum > 0; --layerNum){ //La capa de entrada no tiene pesos
        Layer &layer = m_layers[layerNum];
        Layer &prevLayer = m_layers[layerNum -1];

        for(unsigned n = 0; n < layer.size() -1; ++n){
            layer[n].updateInputWeights(prevLayer);
        }
    }

}

void Net::feedForward(const vector<double> & inputVals){
    cout<<inputVals.size()<< ","<<m_layers[0].size()-1<<endl;
    assert(inputVals.size() == m_layers[0].size() -1);        //Si el argumento es falso, detiene la ejecución.
                                                            //En este caso, se revisa que el número de valores de entrada sea igual al número de neuronas en la capa.
    //Asignar el valor de entrada a las neuronas de entrada
    for(unsigned i = 0; i < inputVals.size(); ++i){
        m_layers[0][i].setOutputVal(inputVals[i]);                       //El valor lo determina la función setOutptVal()
    }

    //Forward propagation
    for(unsigned layerNum = 1; layerNum < m_layers.size(); ++layerNum){
        Layer &prevLayer = m_layers[layerNum - 1];
        for(unsigned n = 0; n < m_layers[layerNum].size() - 1; ++n){
            m_layers[layerNum][n].feedForward(prevLayer);              //La función feedForward de la clase Neuron se encargará de la matemática
        }
    }
}

//Constructor Net
Net::Net(const vector<unsigned> &topology){
    unsigned numLayers = topology.size();       //El tamaño de topology determina el número de capas
    for(unsigned layerNum = 0; layerNum < numLayers; ++layerNum){
        m_layers.push_back(Layer());                  //push_back() permite agregar un nuevo dato al final de una colección, capas en este caso
        unsigned numOutputs = layerNum == topology.size() - 1 ? 0 : topology[layerNum +1 ];
        //Para la nueva capa, se añaden las neuronas y el bias:
        for (unsigned neuronNum = 0; neuronNum <=topology[layerNum]; ++neuronNum){
            m_layers.back().push_back(Neuron(numOutputs, neuronNum));          //back() regresa una referencia a último elemento de un vector
            cout <<"Se creó una neurona " << endl;
        }
        //Forzar el valor de 10 para bias, la última neurona creada
        m_layers.back().back().setOutputVal(1.0);
    }
}


int main(){

   // TrainingData trainData("/tmp/trainingData.txt")
    vector<unsigned> topology;      //Por ejemplo {3,2,1}
    topology.push_back(3);
    topology.push_back(2);
    topology.push_back(1);

    Net myNet(topology);             //topology especifica la arquitectura de la red

    vector<double> inputVals;       //Arreglo de longitud variable
    myNet.feedForward(inputVals);   //Etapa de entrenamiento

    vector<double> targetVals;
    myNet.backprop(targetVals);     //Etapa de aprendizaje

    vector<double> resultVals;
    myNet.getResults(resultVals);   //Salidas


    return 0;

}

