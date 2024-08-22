import { BadRequestException, Injectable, Logger } from '@nestjs/common';
import * as tf from '@tensorflow/tfjs-node';
import * as fs from 'fs';
import * as path from 'path';

@Injectable()
export class ImageClassificationService {
    private model: tf.LayersModel;
    private readonly modelDir = path.resolve(process.cwd(), 'static', 'image-model');
    private readonly logger = new Logger('ImageClassificationService');

    constructor() {
        // Inicializar el modelo en el constructor si es necesario
    }

    private createModel(): tf.Sequential {
        const model = tf.sequential();

        // Primera capa convolucional
        model.add(tf.layers.conv2d({
            inputShape: [128, 128, 3],  // Tamaño de las imágenes de entrada
            filters: 32,                // Número de filtros
            kernelSize: 3,              // Tamaño del kernel (3x3)
            activation: 'relu',         // Función de activación
        }));

        // Primera capa de pooling
        model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

        // Segunda capa convolucional
        model.add(tf.layers.conv2d({
            filters: 64,
            kernelSize: 3,
            activation: 'relu',
        }));

        // Segunda capa de pooling
        model.add(tf.layers.maxPooling2d({ poolSize: [2, 2] }));

        // Capa de flatten para aplanar las características en una dimensión
        model.add(tf.layers.flatten());

        // Capa densa completamente conectada
        model.add(tf.layers.dense({
            units: 128,
            activation: 'relu',
        }));

        // Capa de salida con activación softmax para la clasificación de dos clases
        model.add(tf.layers.dense({
            units: 2,                   // Número de clases (por ejemplo, 'Gato' y 'No Gato')
            activation: 'softmax',      // Softmax para clasificación multiclase
        }));

        // Compilar el modelo
        model.compile({
            optimizer: 'adam',          // Optimizador
            loss: 'categoricalCrossentropy',  // Función de pérdida
            metrics: ['accuracy'],      // Métricas para seguimiento
        });

        return model;
    }

    private async loadImagesFromBuffers(imageBuffers: Buffer[]): Promise<tf.Tensor[]> {
        const tensors = [];
        for (const buffer of imageBuffers) {
            if (!buffer || buffer.length === 0) {
                throw new Error('El buffer de la imagen está vacío o no es válido.');
            }
    
            const imageTensor = tf.node.decodeImage(buffer, 3) // El segundo parámetro '3' fuerza a que solo haya 3 canales (RGB)
                .resizeNearestNeighbor([128, 128])  // Redimensionar a 128x128 píxeles
                .toFloat()
                .div(tf.scalar(255.0))  // Normalizar a [0, 1]
                .expandDims(0);  // Añadir la dimensión del batch
    
            tensors.push(imageTensor);
        }
        return tensors;
    }


    private async loadLabels(labels: number[]): Promise<tf.Tensor> {
        return tf.tensor2d(labels.map(label => [label === 0 ? 1 : 0, label === 1 ? 1 : 0]));
    }

    async trainModel(imageBuffers: Buffer[], labels: number[]): Promise<void> {
        // console.log({imageBuffers})
        // console.log({labels})
        if (!fs.existsSync(this.modelDir)) {
            fs.mkdirSync(this.modelDir, { recursive: true });
        }
    
        // Verificar si el modelo ya está compilado
        if (!this.model) {
            this.model = this.createModel();
        }
    
        // Asegúrate de que el modelo esté compilado antes de entrenarlo
        if (!this.model.optimizer) {
            this.model.compile({
                optimizer: 'adam',
                loss: 'categoricalCrossentropy',
                metrics: ['accuracy'],
            });
            this.logger.log('Modelo compilado.');
        }
    
        const imageTensors = await this.loadImagesFromBuffers(imageBuffers);
        const labelTensor = await this.loadLabels(labels);
    
        const xs = tf.concat(imageTensors);
        const ys = labelTensor;
    
        await this.model.fit(xs, ys, {
            epochs: 500,
            batchSize: 32,
            validationSplit: 0.2,
            callbacks: {
                onEpochEnd: (epoch, logs) => {
                    this.logger.log(`Epoch ${epoch + 1}: Loss = ${logs.loss}, Accuracy = ${logs.acc}`);
                },
            },
        });
    
        await this.model.save(`file://${this.modelDir}`);
        this.logger.log('Modelo entrenado y guardado.');
    }
    async classifyImage(imageBuffer: Buffer) {

        if (!imageBuffer) {
            throw new BadRequestException('No se ha proporcionado una imagen válida');
        }
    
        if (!this.model) {
            await this.loadOrCreateModel();
        }
    
        const imageTensor = tf.node.decodeImage(imageBuffer)
            .resizeNearestNeighbor([128, 128])
            .toFloat()
            .div(tf.scalar(255.0))
            .expandDims(0);
    
        console.log('Image Tensor Shape:', imageTensor.shape);
    
        const prediction = this.model.predict(imageTensor) as tf.Tensor;
        const predictionArray = await prediction.array() as number[][];
    
        console.log('Prediction Array:', predictionArray);
    
        if (predictionArray[0].length < 2) {
            throw new Error('El modelo no está generando dos salidas como se esperaba.');
        }
    
        const gatoProbabilidad = predictionArray[0][1];
        const noGatoProbabilidad = predictionArray[0][0];
        
        const predictedClass = gatoProbabilidad > noGatoProbabilidad ? 'Gato' : 'No Gato';
        const porcentaje = Math.max(gatoProbabilidad, noGatoProbabilidad);
    
        return {
            porcentaje: porcentaje,
            conclusion: predictedClass,
            gatoProbabilidad,
            noGatoProbabilidad
        };
    }

    async loadOrCreateModel(): Promise<void> {
        const modelJsonPath = path.resolve(this.modelDir, 'model.json');

        if (!fs.existsSync(modelJsonPath)) throw new BadRequestException(`primero cargar-entrenar el modelo`)
        //     this.logger.log('No se encontró un modelo existente. Entrenando uno nuevo.');
        //     this.model = this.createModel();
        //     // Opcionalmente, entrena el modelo aquí si es necesario
        // } else {
            this.logger.log(`Cargando el modelo desde: ${modelJsonPath}`);
            this.model = await tf.loadLayersModel(`file://${modelJsonPath}`);
        // }
    }
}
