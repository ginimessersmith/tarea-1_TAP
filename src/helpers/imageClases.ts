type ImageNetClasses = {
    [key: number]: string;
};

const IMAGENET_CLASSES: ImageNetClasses = {
    0: 'tench',
    1: 'goldfish',
    2: 'great white shark',
    3: 'tiger shark',
    // ...
    235: 'tabby cat',
    174: 'sports car',
    // ...
    797: 'toilet tissue',
    // ...
};

export default IMAGENET_CLASSES;