import express from "express";
import { InferenceSession, Tensor} from 'onnxruntime-node';
import bodyParser from "body-parser";
import { Jimp } from "jimp";

const IMAGE_SIZE = [640, 640];
const RGB_MEAN = [0.485, 0.456, 0.406];
const RGB_STD = [0.229, 0.224, 0.225];

const app = express();
app.use(bodyParser.raw({type: "image/png", limit: '50mb'}));

let session: InferenceSession;

async function load_model(model_path: string) {
    const session = await InferenceSession.create(
        model_path,
        {
            "executionProviders": ["webgpu"]
        }
    )
    return session;
}

session = await load_model("D:/Hobbies/projects/shoreline-remote-sensing/weights/dinov3-vits16/model.onnx");

async function prepare_input(input_data: Buffer){
    const jimp_image = await Jimp.fromBuffer(input_data);
    const resized_image = jimp_image.resize({
        w: IMAGE_SIZE[0],
        h: IMAGE_SIZE[1]
    })
    const imagebufferdata = resized_image.bitmap.data;
    const [redArray, greenArray, blueArray] = new Array(new Array<number>(), new Array<number>(), new Array<number>());
    for (let i = 0; i < imagebufferdata.length; i += 4) {
        redArray.push((imagebufferdata[i]/255.0 - RGB_MEAN[0])/RGB_STD[0]);
        greenArray.push((imagebufferdata[i+1]/255.0 - RGB_MEAN[1])/RGB_STD[1]);
        blueArray.push((imagebufferdata[i+2]/255.0 - RGB_MEAN[2])/RGB_STD[2]);
        // skip data[i + 3] to filter out the alpha channel
    }
    const transposedData = redArray.concat(greenArray).concat(blueArray);
    let l = transposedData.length;
    const float32Data = new Float32Array(3 * IMAGE_SIZE[0] * IMAGE_SIZE[1]);
    for(let i = 0; i < l; i++){
        float32Data[i] = transposedData[i];
    }

    const inputTensor = new Tensor("float32", float32Data, [1, 3, IMAGE_SIZE[1], IMAGE_SIZE[0]]);
    return inputTensor;
}

async function run_inference(session: InferenceSession, input: InferenceSession.FeedsType){
    const results = await session.run(input);
    return results;
}

app.post("/predict", async(req, res)=>{
    const imagebuffer: Buffer = req.body;
    const prepared_input = await prepare_input(imagebuffer);
    const input_name = session.inputNames[0];
    const feeds: Record<string, Tensor> = {};
    feeds[input_name] = prepared_input;
    const inferenceResult = await run_inference(session, feeds);
    const result = inferenceResult["pooler_output"];
    console.log(result);
    const embeddings = await result.getData();
    res.status(201).send({embeddings: Array.prototype.slice.call(embeddings)});
});

const args = process.argv.slice(2);
let port: number;

if(args.length == 0){
    console.log("NO PORTS SPECIFIED! Using default 8000");
    port = 8000;
}else{
    port = parseInt(args[0]);
}

const server = app.listen(port, () =>
  console.log(`Server ready at: http://localhost:${port}`)
);
