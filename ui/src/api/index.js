// import axios from 'axios';

// const BASE_URL = process.env.REACT_APP_API_URL;

// const client = axios.create({
//   baseURL: BASE_URL,
//   json: true,
// });

export const getDataResult = async id => {
  // TODO(tudor): remove mock
  // return client.get(`/data/${id}`);

  return {
    x: 'lorem ipsum',
  };
};

export const uploadData = async file => {
  const formData = new FormData();
  formData.append('file', file);

  // TODO(tudor): remove mock
  // return client.post('/data', formData, {
  //   headers: {
  //     'Content-Type': 'multipart/form-data',
  //   },
  // });

  return { id: '1234' };
};
