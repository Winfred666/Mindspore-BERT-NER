// for global event emitter
import mitt from 'mitt';

export const emitter = mitt();


// color for avatar
function getRandomVividColor() {
    // Generate random hue (0 to 360 degrees)
    const hue = Math.floor(Math.random() * 360);

    // Saturation and lightness are kept high for vivid colors
    const saturation = 60 + Math.random() * 30; // Between 70% and 100%
    const lightness = 40 + Math.random() * 10; // Between 50% and 70%

    // Return the HSL color as a string
    return `hsl(${hue}, ${saturation}%, ${lightness}%)`;
}

let avatarColorMap = {}

// clear color map, means a new dialog is created
export const clearColorMap = () => {
    avatarColorMap = {};
}

export const getColor = (name) =>{
    if (!avatarColorMap[name]) {
        avatarColorMap[name] = getRandomVividColor();
    }
    return avatarColorMap[name];
}