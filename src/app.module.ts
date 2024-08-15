import { Module } from '@nestjs/common';

import { TensorflowModule } from './tensorflow/tensorflow.module';
import { ConfigModule } from '@nestjs/config';
import { ImageClassificationService } from './image-classification/image-classification.service';

@Module({
  imports: [
    ConfigModule.forRoot({
      isGlobal: true, 
    }),
    TensorflowModule],
  controllers: [],
  providers: [ImageClassificationService],
})
export class AppModule {}
