import { Injectable, Logger } from '@nestjs/common';
import { CreateTensorflowDto } from './dto/create-tensorflow.dto';
import { UpdateTensorflowDto } from './dto/update-tensorflow.dto';
import * as tf from '@tensorflow/tfjs-node';
import * as path from 'path';
import * as fs from 'fs';


@Injectable()
export class TensorflowService {

  private model: tf.Sequential;
  private modelDir = path.resolve(process.cwd(), 'static', 'model');
  private logger = new Logger('tensorflow')

  constructor() {
    // this.loadModel();
  }

  // async loadModel() {
  //   this.model = await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
  // }

  // performBasicOperation(): string {
  //   // Crear dos tensores y realizar una suma
  //   const tensor1 = tf.tensor([1, 2, 3, 4]);
  //   const tensor2 = tf.tensor([5, 6, 7, 8]);

  //   const sumTensor = tensor1.add(tensor2);

  //   // Convertir el resultado a un array y devolverlo como string
  //   const result = sumTensor.arraySync();
  //   return `Resultado de la suma de los tensores: ${result}`;
  // }

  // async processImage(imageBuffer: Buffer) {
  //   // Cargar la imagen desde un buffer
  //   const imageTensor = tf.node.decodeImage(imageBuffer, 3);

  //   // Cambiar el tamaño de la imagen a 224x224
  //   const resizedImage = tf.image.resizeBilinear(imageTensor, [224, 224]);

  //   // Obtener las dimensiones de la imagen
  //   const shape = resizedImage.shape;

  //   // Convertir la imagen a un array
  //   const imageArray = resizedImage.arraySync();

  //   return {
  //     shape,
  //     imageArraySnippet: imageArray.slice(0, 5) // Solo un fragmento de la matriz para la vista previa
  //   };
  // }

  //* Método para clasificar una imagen
  // async classifyImage(imageBuffer: Buffer): Promise<string[]> {
  //   // Decodificar la imagen
  //   const imageTensor = tf.node.decodeImage(imageBuffer, 3)
  //     .resizeNearestNeighbor([224, 224]) // Redimensionar la imagen a 224x224
  //     .expandDims() // Añadir una dimensión adicional (batch size)
  //     .toFloat()
  //     .div(tf.scalar(127.5)) // Normalizar los valores
  //     .sub(tf.scalar(1));

  //   // Hacer la predicción
  //   const predictions = this.model.predict(imageTensor) as tf.Tensor;
  //   const predictionArray = (await predictions.array()) as number[][];

  //   // Extraer las etiquetas y devolverlas
  //   const topPredictions = this.getTopKPredictions(predictionArray, 5);
  //   return topPredictions;
  // }

  // // Método para obtener las mejores predicciones
  // private getTopKPredictions(predictions: number[][], k: number): string[] {
  //   const topK = predictions[0] // Usamos la primera fila si es un batch único
  //   .map((score, i) => ({ score, label: `Class ${i}` })) // Aquí deberías mapear a las etiquetas reales
  //   .sort((a, b) => b.score - a.score)
  //   .slice(0, k);

  // return topK.map(p => `${p.label}: ${p.score.toFixed(4)}`);
  // }

  async loadOrCreateModel(): Promise<void> {
    // Asegurarte de que el directorio existe
    if (!fs.existsSync(this.modelDir)) {
      fs.mkdirSync(this.modelDir, { recursive: true });
      this.logger.log(`Directorio creado: ${this.modelDir}`);
      this.createNewSequentialModel();
      return;
    }

    const modelJsonPath = path.resolve(this.modelDir, 'model.json');

    if (fs.existsSync(modelJsonPath)) {
      this.logger.log(`Cargando el modelo desde: ${modelJsonPath}`);
      try {
        const loadedModel = await tf.loadLayersModel(`file://${modelJsonPath}`);
        if (loadedModel instanceof tf.Sequential) {
          this.model = loadedModel;
          this.logger.log('Modelo secuencial cargado desde archivo.');

          // Compilar el modelo después de cargarlo
          this.model.compile({
            optimizer: 'sgd',
            loss: 'meanSquaredError',
          });
        } else {
          this.logger.error('El modelo cargado no es secuencial. Se creará un nuevo modelo secuencial.');
          this.createNewSequentialModel();
        }
      } catch (error) {
        this.logger.error(`Error al cargar el modelo: ${error}`);
        this.createNewSequentialModel();
      }
    } else {
      this.logger.log('No se encontró un modelo existente. Creando uno nuevo.');
      this.createNewSequentialModel();
    }
  }

  private createNewSequentialModel(): void {
     this.model = tf.sequential();
    this.model.add(tf.layers.dense({ inputShape: [1], units: 1 }));
  
    this.model.compile({
      optimizer: tf.train.sgd(0.001),  // Tasa de aprendizaje aún más baja
      loss: 'meanSquaredError',
    });
    this.logger.log('Nuevo modelo secuencial creado.');
  }

  async trainModel(fahrenheitValues: number[], celsiusValues: number[]): Promise<void> {
    if (!fs.existsSync(this.modelDir)) {
      fs.mkdirSync(this.modelDir, { recursive: true });
      this.logger.log(`Directorio creado para guardar el modelo: ${this.modelDir}`);
    }
  
    // Verificar los datos antes de crear tensores
    if (fahrenheitValues.some(v => v == null || isNaN(v)) || celsiusValues.some(v => v == null || isNaN(v))) {
      throw new Error('Los datos de entrenamiento contienen valores no válidos.');
    }
  
    const fahrenheitTensor = tf.tensor1d(fahrenheitValues).div(tf.scalar(100));
    const celsiusTensor = tf.tensor1d(celsiusValues).div(tf.scalar(100));
  
    // Imprimir los pesos antes del entrenamiento
    this.logger.log('Pesos antes del entrenamiento:');
    this.model.layers.forEach(layer => {
      const weights = layer.getWeights();
      weights.forEach(weight => {
        this.logger.log(weight.arraySync());
      });
    });
  
    await this.model.fit(fahrenheitTensor, celsiusTensor, {
      epochs: 500,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          this.logger.log(`Epoch ${epoch}: loss = ${logs?.loss}`);
        },
      },
    });
  
    // Imprimir los pesos después del entrenamiento
    this.logger.log('Pesos después del entrenamiento:');
    this.model.layers.forEach(layer => {
      const weights = layer.getWeights();
      weights.forEach(weight => {
        this.logger.log(weight.arraySync());
      });
    });
  
    await this.model.save(`file://${this.modelDir}`);
    this.logger.log('Modelo guardado correctamente.');
  }

  async predictCelsius(fahrenheitValue: number) {
    if (!this.model) {
      await this.loadOrCreateModel();
    }

    // Normalizar el valor de entrada durante la predicción
    const fahrenheitTensor = tf.tensor1d([fahrenheitValue]).div(tf.scalar(100));

    // Realiza la predicción
    const celsiusTensor = this.model.predict(fahrenheitTensor) as tf.Tensor;
    const celsiusArray = await celsiusTensor.array() as number[][];
    const celsiusValue = celsiusArray.length && celsiusArray[0].length ? celsiusArray[0][0] : null;

    // Desnormalizar el resultado
    const desnormalizedCelsiusValue = celsiusValue !== null ? celsiusValue * 100 : null;

    this.logger.log(`Valor Celsius predicho: ${desnormalizedCelsiusValue}`);

    return desnormalizedCelsiusValue;
  }
}
