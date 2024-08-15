import { PartialType } from '@nestjs/mapped-types';
import { CreateTensorflowDto } from './create-tensorflow.dto';

export class UpdateTensorflowDto extends PartialType(CreateTensorflowDto) {}
