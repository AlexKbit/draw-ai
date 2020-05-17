var canvas, ctx;
var flag = false;
var prevX = 0;
var currX = 0;
var prevY = 0;
var currY = 0;
var dot_flag = false;

var lineColor = "black";
var lineWidth = 3;

function init() {
    canvas = document.getElementById('canvas');
    ctx = canvas.getContext("2d");
    w = canvas.width;
    h = canvas.height;

    canvas.addEventListener("mousemove", function (e) {
        findxy('move', e)
    }, false);
    canvas.addEventListener("mousedown", function (e) {
        findxy('down', e)
    }, false);
    canvas.addEventListener("mouseup", function (e) {
        findxy('up', e)
    }, false);
    canvas.addEventListener("mouseout", function (e) {
        findxy('out', e)
    }, false);

    document.getElementById('downloadBtn').addEventListener('click', function() {
        downloadCanvas(this, 'img_' + generateId() + '.png');
    }, false);
}

function draw() {
    ctx.beginPath();
    ctx.moveTo(prevX, prevY);
    ctx.lineTo(currX, currY);
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = lineWidth;
    ctx.stroke();
    ctx.closePath();
}

function erase() {
    ctx.clearRect(0, 0, w, h);
    document.getElementById('img_class').innerHTML = 'Class: ?';
}

function findxy(res, e) {
    if (res == 'down') {
        prevX = currX;
        prevY = currY;
        currX = e.clientX - canvas.offsetLeft;
        currY = e.clientY - canvas.offsetTop;

        flag = true;
        dot_flag = true;
        if (dot_flag) {
            ctx.beginPath();
            ctx.fillStyle = lineColor;
            ctx.fillRect(currX, currY, 2, 2);
            ctx.closePath();
            dot_flag = false;
        }
    }
    if (res == 'up' || res == "out") {
        flag = false;
    }
    if (res == 'move') {
        if (flag) {
            prevX = currX;
            prevY = currY;
            currX = e.clientX - canvas.offsetLeft;
            currY = e.clientY - canvas.offsetTop;
            draw();
        }
    }
}

function downloadCanvas(link, filename) {
    link.href = document.getElementById('canvas').toDataURL("image/png");
    link.download = filename;
}

function generateId() {
    return Math.random().toString(16).substring(2);
}

function submitImage() {
    var canvasData = canvas.toDataURL();

    $.ajax({
        type: "POST",
        url: "/classify",
        data: {image: canvasData},
        dataType: 'json',
        success: function(result){
            console.info(result);
            document.getElementById('img_class').innerHTML = 'Class: ' + result.label_name;
        }
    }).done(function() {
        console.log('Sent image');
    });
}

$('document').ready(init);
