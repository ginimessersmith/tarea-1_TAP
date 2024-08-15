import { Module } from '@nestjs/common';
import { TensorflowService } from './tensorflow.service';
import { TensorflowController } from './tensorflow.controller';
import { MulterModule } from '@nestjs/platform-express';
import { ImageClassificationService } from 'src/image-classification/image-classification.service';

@Module({
  imports: [
    MulterModule.register({
      dest: './uploads',  // Directorio para guardar las imágenes subidas temporalmente
    }),
    
  ],
  controllers: [TensorflowController],
  providers: [TensorflowService,ImageClassificationService],
})
export class TensorflowModule {}
