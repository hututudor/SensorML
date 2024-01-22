const BASE_URL = process.env.REACT_APP_API_URL;

export const Graph = ({ name, model }) => (
  <iframe
    src={`${BASE_URL}/${model}/${name}.html`}
    title={`${model}/${name}`}
    width='1400px'
    height='500px'
  />
);
