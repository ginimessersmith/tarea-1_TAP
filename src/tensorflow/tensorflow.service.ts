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
    // this.createCNNModel();
    this.createDenseModel();
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
    if (!fs.existsSync(this.modelDir)) {
      fs.mkdirSync(this.modelDir, { recursive: true });
      this.logger.log(`Directorio creado: ${this.modelDir}`);
      this.createDenseModel();  
      return;
    }

    const modelJsonPath = path.resolve(this.modelDir, 'model.json');

    if (fs.existsSync(modelJsonPath)) {
      this.logger.log(`Cargando el modelo desde: ${modelJsonPath}`);
      try {
        const loadedModel = await tf.loadLayersModel(`file://${modelJsonPath}`);
        this.model = loadedModel as tf.Sequential;  // Asegúrate de que sea secuencial

        this.model.compile({
          optimizer: tf.train.sgd(0.0001),  // Tasa de aprendizaje para CNN
          loss: 'meanSquaredError',
        });
        this.logger.log('Modelo CNN cargado desde archivo.');
      } catch (error) {
        this.logger.error(`Error al cargar el modelo: ${error}`);
        this.createDenseModel();  // Reemplaza la lógica secuencial con CNN
      }
    } else {
      this.logger.log('No se encontró un modelo existente. Creando uno nuevo.');
      this.createDenseModel();
    }
  }
  // * -------------------------------------------------------------------------------------------
  private createDenseModel(): void {
    const model = tf.sequential();

    model.add(tf.layers.dense({
      inputShape: [1],
      units: 1,
      kernelInitializer: 'glorotUniform',  // Inicialización estándar
      activation: 'linear'
  }));

    model.compile({
      optimizer: tf.train.sgd(0.0001),
      loss: 'meanSquaredError'
    });

    this.model = model;
    this.logger.log('Nuevo modelo denso creado.');
  }

  async loadOrCreateModelDense(): Promise<void> {
    if (!fs.existsSync(this.modelDir)) {
      fs.mkdirSync(this.modelDir, { recursive: true });
      this.logger.log(`Directorio creado: ${this.modelDir}`);
      this.createDenseModel();  // Crea un nuevo modelo denso si no existe ninguno
      return;
    }

    const modelJsonPath = path.resolve(this.modelDir, 'model.json');

    if (fs.existsSync(modelJsonPath)) {
      this.logger.log(`Cargando el modelo desde: ${modelJsonPath}`);
      try {
        const loadedModel = await tf.loadLayersModel(`file://${modelJsonPath}`);
        this.model = loadedModel as tf.Sequential;

        this.model.compile({
          optimizer: tf.train.sgd(0.001),
          loss: 'meanSquaredError',
        });
        this.logger.log('Modelo denso cargado desde archivo.');
      } catch (error) {
        this.logger.error(`Error al cargar el modelo: ${error}`);
        this.createDenseModel();  // Crea un nuevo modelo si hay un error al cargar
      }
    } else {
      this.logger.log('No se encontró un modelo existente. Creando uno nuevo.');
      this.createDenseModel();
    }
  }

  async trainModelDense(fahrenheitValues: number[], celsiusValues: number[]): Promise<void> {
    if (!fs.existsSync(this.modelDir)) {
      fs.mkdirSync(this.modelDir, { recursive: true });
      this.logger.log(`Directorio creado para guardar el modelo: ${this.modelDir}`);
    }
  
    if (fahrenheitValues.some(v => v == null || isNaN(v)) || celsiusValues.some(v => v == null || isNaN(v))) {
      throw new Error('Los datos de entrenamiento contienen valores no válidos.');
    }
  
    // Normalización
    const fahrenheitTensor = tf.tensor1d(fahrenheitValues).div(tf.scalar(100));
    const celsiusTensor = tf.tensor1d(celsiusValues).div(tf.scalar(100));
  
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

  async predictCelsiusDense(fahrenheitValue: number) {
    if (!this.model) {
      await this.loadOrCreateModel();
    }
  
    // Normalización del valor de entrada
    const fahrenheitTensor = tf.tensor1d([fahrenheitValue]).div(tf.scalar(100));
  
    const celsiusTensor = this.model.predict(fahrenheitTensor) as tf.Tensor;
    const celsiusArray = await celsiusTensor.array() as number[];
    let celsiusValue = celsiusArray.length ? celsiusArray[0] : null;
  
    // Desnormalización del valor de salida
    if (celsiusValue !== null) {
      celsiusValue = celsiusValue * 100;
    }
  
    this.logger.log(`Valor Celsius predicho: ${celsiusValue}`);
  
    return celsiusValue;
  }
}
