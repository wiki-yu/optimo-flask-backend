const express = require('express');
const fileUpload = require('express-fileupload');
const cors = require('cors');
const fs = require('fs');
var parse = require('csv-parse')

const port = 4500
const app = express();
const {spawn} = require('child_process');

// middle ware
app.use(express.static('public'))
app.use(cors())
app.use(fileUpload());

var videoDetection = (req, res) => {
    console.log('[INFO]Backend Video detection prcoess started!');
    const myFile = req.files.file;
    console.log(myFile.name)
    console.timeEnd('timeCal1')
    console.time('timeCal2')
    const python = spawn('python', ['./src/adabegi.py', '--video_name', myFile.name]);

    python.stdout.on('data', function (data) {
        console.log('[INFO]Pipe data from python image_detection script ...' + data);
        console.timeEnd('timeCal2')
        console.time('timeCal3')
        // res.json({serverUrl: `${dataUrl}`, imgInfo: JSON.parse(data.toString())});
        fs.readFile(
            './outputs/' + myFile.name, 'base64', 
            (err, base64Image) => {
                const detectedVideoUrl = `data:video/mp4;base64, ${base64Image}`
                return res.send(`${detectedVideoUrl}`);
            }
        )
        // csv_name = "CoG_with_task_" + myFile.name + ".csv"
        // fs.readFile(
        //     './outputs/' + csv_name, 'base64', 
        //      (err, fileData) => {
        //         parse(fileData, {columns: false, trim: true}, function(err, rows) {
        //         console.log("fs read csv file test.....")
        //         console.timeEnd('timeCal3')
            
        //     })
        //   })
        // res.json(`${detectedImgUrl}`);
    });
}


app.use(function(req, res, next) {
    res.header("Access-Control-Allow-Origin", "*");
    res.header("Access-Control-Allow-Headers", "Origin, X-Requested-With, Content-Type, Accept");
    next();
  });


app.post('/motionDetection', (req, res) => {
    console.log("[INFO]Get ready to recieve VIDEO DETECTION files!")
    console.time('timeCal1');
    if (!req.files) {
        return res.status(500).send({ msg: "[ERR]File not found" })
    }

    const myFile = req.files.file;
    console.log("[INFO]Receiving Detection video files!")
    console.log("[INFO]File name:", myFile.name)

    // Use the mv() method to place the file somewhere on your server
    myFile.mv(`${__dirname}/data/${myFile.name}`, function (err) {
        console.log('[INFO]mv callback')
        if (err) {
            console.log(err)
            return res.status(500).send({ msg: "[ERR]there is error" });
        }
        fs.readFile(
            './data/' + myFile.name, 'base64', 
            (err, base64Image) => {
                const dataUrl = `data:video/mp4;base64, ${base64Image}`
                // console.log("received video", dataUrl)
                videoDetection(req, res);
                // return res.send(`${dataUrl}`);
            }
        )
    });
})

app.listen(5000, () => {
    console.log('server is listening at port 5000');
})