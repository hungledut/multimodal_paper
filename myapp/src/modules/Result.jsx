import React, { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { useLocation } from "react-router-dom";
import Progressbar from "../components/Progressbar/Progressbar";

const Result = () => {
  const state = useLocation();
  const [rs, setRs] = useState([]);
  const { text, result, temp } = state.state;
  const modelResults = [
    result.results_mlp,
    result.results_rnn,
    result.results_lstm,
    result.results_convnext,
    result.results_efficientnetb0,
    result.results_mobilenetv2,
    result.results_resnet18,
    result.results_vision,
    result.results_vit,
    result.results_multimodal,
    result.results_proposed_multimodal,
  ];
  const nameModel = [
    "MLP",
    "RNN",
    "LSTM",
    "ConvNext",
    "EfficientNetB0",
    "MobileNetV2",
    "ResNet18",
    "VGG",
    "ViT",
    "Multimodal",
    "Proposed Multimodal",
  ]
  let arrays = [];
  let t = 0;
  // Tạo mảng con rỗng cho mỗi item
  for (let i = 0; i < temp; i++) {
    arrays.push(new Set());
  }
  useEffect(() => {
    async function handle() {
      // Duyệt qua từng model
      modelResults.forEach((modelResult, modelIndex) => {
        // Duyệt qua từng item trong model và đưa vào mảng tương ứng
        modelResult.forEach(async (item, index) => {
          await arrays[index].add(item);
          arrays[index] = Array.from(arrays[index]);
        });
      });
    }
    handle();
    setTimeout(() => {
      arrays = Array.from(arrays);
      arrays = arrays.map(set => Array.from(set));
      console.log(arrays[0][2][2]);
      setRs(arrays);
    }, 1000);
    console.log(result);
  }, [result, temp]);

  return (
    <div className="flex flex-col gap-5 py-14 px-6">
      <div className="flex items-end justify-center">
        <Link to={"/"}>
          <button className="btnback">Back</button>
        </Link>
      </div>
      <div className="w-full flex flex-col gap-y-[30px] items-center justify-center border border-white rounded-xl p-[30px]">
        {rs.map((item, index) => (
          <div key={index} className="mb-[30px] flex items-center justify-center gap-[200px]">
            <div class="card">
              <div class="card-image">
                <img src={item[3][2]} alt="" class="image1"></img>
              </div>
              <p class="card-title">Description</p>
              <p class="card-body">
                {text[index]}
              </p>
            </div>
            <div className="flex flex-col items-center justify-between gap-x-[50px] gap-y-[15px]">
              <div className="flex w-full items-center justify-between gap-[120px] font-bold">
                <div className="flex items-center gap-[20px]">
                  <span><strong className="text-[20px] text-pink-300">Result</strong></span>
                </div>
                <div className="flex items-center gap-[20px] -translate-x-[150px]">
                  <span><strong className="text-[20px] text-pink-300">Model</strong></span>
                </div>
                <div className="flex items-center gap-[20px] -translate-x-[130px]">
                  <span><strong className="text-[20px] text-pink-300">Score</strong></span>
                </div>
              </div>

              {item.map((item1, index1) => (
                <div className="flex w-full items-center justify-between gap-[50px]" key={index1}>
                  <div className="flex items-center gap-[20px]">
                    <span><strong>{item1[0]}</strong></span>
                  </div>
                  <div className="flex items-center gap-[20px]">
                    <span><strong>{nameModel[index1]}</strong></span>
                  </div>
                  <div className="flex items-center gap-[20px]">
                    <Progressbar percent={item1[1]} />
                  </div>
                </div>
              ))}
            </div>
          </div>
        ))}
      </div>

    </div>
  );
};
// const Result = () => {
//   const state = useLocation();
//   const [rs, setRs] = useState([]);
//   const { result, temp } = state.state;
//   const modelResults = [
//     result.results_convnext,
//     result.results_efficientnetb0,
//     result.results_mobilenetv2,
//     result.results_multimodal,
//     result.results_proposed_multimodal,
//     result.results_resnet18,
//     result.results_vision,
//     result.results_vit
//   ];
//   let arrays = [];

//   // Tạo mảng con rỗng cho mỗi item
//   for (let i = 0; i < temp; i++) {
//     arrays.push(new Set());
//   }
//   useEffect(() => {
//     async function handle() {
//       // Duyệt qua từng model
//       modelResults.forEach((modelResult, modelIndex) => {
//         // Duyệt qua từng item trong model và đưa vào mảng tương ứng
//         modelResult.forEach(async (item, index) => {
//           await arrays[index].add(item);
//           arrays[index] = Array.from(arrays[index]);
//         });
//       });
//     }
//     handle();
//     setTimeout(() => {
//       arrays = Array.from(arrays);
//       arrays = arrays.map(set => Array.from(set));
//       console.log(arrays[0][2][2]);
//       setRs(arrays);
//     }, 1000);
//     // Lúc này, arrays sẽ chứa temp mảng con, mỗi mảng con chứa item từng model tương ứng

//   }, [result, temp]);

//   return (
//     <>
//      {rs.map((item, index) => (
//       <div className="text-white" key={index}>
//         <img src={item[0][2]} alt="" />
//         {item.map((item1, index1) => (
//           <div key={index1}>
//             <div className="">{item1[0]}</div>
//             <div className="">{item1[1]}</div>
//           </div>
//         ))}
//       </div>
//     ))}
//     </>
//   );
// };
export default Result;
