var canvas = null;
var ctx = null;
var labels = ["apple", "clock", "door", "diamond", "fork", "eye", "star", "axe", "sword", "fish"];
var targetLabel = '?';
var resultLabel = '?';
var imdData = null;
var iterationStep = 0;

function init() {
    canvas = document.createElement("canvas");
    canvas.width = document.getElementById('ganId').width;
    canvas.height = document.getElementById('ganId').height;
    ctx = canvas.getContext('2d');
}

$('document').ready(init);

function generateImgUrl(label) {
    return window.location.href+'/../generate/' + label + '?rnd=' + generateId();
}

function setRandomLabel() {
    label = labels[Math.floor(Math.random() * labels.length)];
    targetLabel = label;
    refreshLabels();
}

function generateId() {
    return Math.random().toString(16).substring(2);
}

function submitImage() {
    $.ajax({
        type: "POST",
        url: "/classify",
        data: {image: imdData},
        dataType: 'json',
        success: function(result){
            console.info(result);
            resultLabel = result.label_name;
            refreshLabels();
        }
    }).done(function() {
        console.log('Sent image');
    });
}

function toDataURL(img) {
    ctx.drawImage(img, 0, 0);
    return canvas.toDataURL();
}

function generateImg() {
    var label = targetLabel
    var img = document.getElementById('ganId');
    img.onload = function () {
        imdData = toDataURL(img);
    };
    img.src = generateImgUrl(label);
}

function refreshLabels() {
    document.getElementById('targetLabelId').innerHTML = targetLabel;
    document.getElementById('resultLabelId').innerHTML = resultLabel;
}

function compareLabels() {
    if (targetLabel == resultLabel) {
        document.getElementById('targetLabelId').style['color'] = 'green';
        document.getElementById('resultLabelId').style['color'] = 'green';
        document.getElementById('result').style['color'] = 'green';
        document.getElementById('result').innerHTML = 'Success';
    } else {
        document.getElementById('targetLabelId').style['color'] = 'red';
        document.getElementById('resultLabelId').style['color'] = 'red';
        document.getElementById('result').style['color'] = 'red';
        document.getElementById('result').innerHTML = 'Failed';
    }
    refreshLabels();
}

function clearState() {
    targetLabel = '?';
    resultLabel = '?';
    document.getElementById('targetLabelId').style['color'] = 'black';
    document.getElementById('resultLabelId').style['color'] = 'black';
    document.getElementById('result').innerHTML = '';
    document.getElementById('ganId').src='/images/nothing.png'
    refreshLabels();
}

function tick() {
    if (iterationStep == 0) {
        setRandomLabel();
        iterationStep = iterationStep + 1;
        return;
    }
    if (iterationStep == 1) {
        generateImg();
        iterationStep = iterationStep + 1;
        return;
    }
    if (iterationStep == 2) {
        submitImage();
        iterationStep = iterationStep + 1;
        return;
    }
    if (iterationStep == 3) {
        compareLabels();
        iterationStep = iterationStep + 1;
        return;
    }
    if (iterationStep == 4) {
        clearState();
        iterationStep = 0;
        return;
    }
    clearState();
    iterationStep = 0;
}

setInterval(() => {
  tick();
}, 1000);