import { Test, TestingModule } from '@nestjs/testing';
import { ImageClassificationService } from './image-classification.service';

describe('ImageClassificationService', () => {
  let service: ImageClassificationService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [ImageClassificationService],
    }).compile();

    service = module.get<ImageClassificationService>(ImageClassificationService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });
});
