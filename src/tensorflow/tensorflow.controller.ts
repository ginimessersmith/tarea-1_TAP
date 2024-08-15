import { Controller, Get, Post, Body, Patch, Param, Delete, UseInterceptors, UploadedFile, UploadedFiles, BadRequestException } from '@nestjs/common';
import { TensorflowService } from './tensorflow.service';
import { CreateTensorflowDto } from './dto/create-tensorflow.dto';
import { UpdateTensorflowDto } from './dto/update-tensorflow.dto';
import { FileFieldsInterceptor, FileInterceptor } from '@nestjs/platform-express';
import { Express } from 'express';
import { ImageClassificationService } from 'src/image-classification/image-classification.service';
import * as multer from 'multer';

@Controller('tensorflow')
export class TensorflowController {
  constructor(
    private readonly tensorflowService: TensorflowService,
    private readonly imageClassificationService: ImageClassificationService,) { }

  // @Post('process')
  // @UseInterceptors(FileInterceptor('image'))
  // async processImage(@UploadedFile() file: Express.Multer.File) {
  //   return this.tensorflowService.processImage(file.buffer);

  // }

  // @Post('classify')
  // @UseInterceptors(FileInterceptor('image'))
  // classifyImage(@UploadedFile() file: Express.Multer.File) {
  //   return this.tensorflowService.classifyImage(file.buffer);

  // }

  // @Get()
  // getHelloWorld(): string {
  //   return this.tensorflowService.performBasicOperation();
  // }

  @Post('train')
  async trainModelFahrenheitCelsius(
    @Body() body: { fahrenheitValues: number[], celsiusValues: number[] }
  ): Promise<string> {
    // Cargar o crear el modelo
    await this.tensorflowService.loadOrCreateModel();

    // Entrenar el modelo con los datos proporcionados
    await this.tensorflowService.trainModel(body.fahrenheitValues, body.celsiusValues);

    return 'Modelo entrenado y guardado exitosamente.';
  }

  @Post('train-imagenes-cats')
  @UseInterceptors(FileFieldsInterceptor([
    { name: 'images', maxCount: 100 }, // Cambiado a FileFieldsInterceptor para manejar múltiples archivos
  ], {
    storage: multer.memoryStorage(), // Almacena en memoria en lugar de en el sistema de archivos
  }))
  async trainModel(
    @UploadedFiles() files: { images?: Express.Multer.File[] },
    @Body('labels') labels: string 
  ) {
   

    if (!files.images || files.images.length === 0) {
      throw new BadRequestException('No se han proporcionado imágenes para entrenar');
    }
    // Convertir las etiquetas de string a array de números
    const labelArray = labels.split(',').map(label => parseInt(label.trim(), 10));
    if (labelArray.length !== files.images.length) {
      throw new BadRequestException('El número de etiquetas debe coincidir con el número de imágenes');
    }

    const imageBuffers = files.images.map(file => file.buffer);
    // console.log({ files });  // Debería mostrar las imágenes recibidas
    console.log({ labelArray }); // Debería mostrar las etiquetas recibidas
    await this.imageClassificationService.trainModel(imageBuffers, labelArray);

    return 'Modelo entrenado correctamente';
  }
  

  @Post('predict')
  async predictCelsius(
    @Body() body: { fahrenheitValue: number }
  ) {
    const celsiusValue = await this.tensorflowService.predictCelsius(body.fahrenheitValue);
    return { celsiusValue };
  }

  @Post('classify-cats')
  @UseInterceptors(FileInterceptor('image', {
    storage: multer.memoryStorage(),
  }))
  classifyImage(
    @UploadedFile() file: Express.Multer.File,
  ) {
    console.log({file})
    return this.imageClassificationService.classifyImage(file.buffer);
  }

}
