var labels = ["apple", "cloud", "diamond", "donut", "door", "mug", "house", "butterfly", "microphone", "bicycle"];

function generateImg() {
    label = getRandomLabel();
    canvas = document.getElementById('showImage').src = window.location.href+'/../generate/' + label + '?rnd=' + generateId();
    document.getElementById('img_class').innerHTML = 'Class: ' + label;
}

function getRandomLabel() {
    return labels[Math.floor(Math.random() * labels.length)];
}

function generateId() {
    return Math.random().toString(16).substring(2);
}