import axios from 'axios';

const BASE_URL = process.env.REACT_APP_API_URL;

const client = axios.create({
  baseURL: BASE_URL,
  json: true,
});

export const getDataResult = async () => {}; // client.get(`/data`);

export const uploadData = async file => {
  const formData = new FormData();
  formData.append('file', file);

  return client.post('/data', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });
};
