import { useState } from 'react';
import axios from 'axios';

function App() {
  const [file, setFile] = useState(null);
  const [method, setMethod] = useState('dehazing');
  const [result, setResult] = useState(null);

  const handleSubmit = async () => {
    const formData = new FormData();
    formData.append('image', file);
    formData.append('method', method);

    const response = await axios.post('http://localhost:5000/process', formData, {
      responseType: 'blob',
    });

    setResult(URL.createObjectURL(response.data));
  };

  return (
    <div className="p-8 max-w-xl mx-auto">
      <h1 className="text-xl font-bold mb-4">图像处理工具</h1>
      <input type="file" onChange={(e) => setFile(e.target.files[0])} />
      <select value={method} onChange={(e) => setMethod(e.target.value)} className="block my-4 p-2">
        <option value="dehazing">去雾</option>
        <option value="deblur">去模糊</option>
        <option value="morphing">形态变换</option>
      </select>
      <button onClick={handleSubmit} className="bg-blue-600 text-white px-4 py-2 rounded">处理</button>
      {result && <img src={result} alt="结果" className="mt-4 border" />}
    </div>
  );
}

export default App;
