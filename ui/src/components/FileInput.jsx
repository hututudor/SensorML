import React from 'react';

export const FileInput = ({ onChange, children, ...rest }) => {
  const fileInputRef = React.useRef();

  const handleButtonClick = () => {
    fileInputRef.current.click();
  };

  return (
    <>
      {React.cloneElement(children, {
        onClick: handleButtonClick,
      })}

      <input
        type='file'
        onChange={onChange}
        ref={fileInputRef}
        style={{ display: 'none' }}
        {...rest}
      />
    </>
  );
};
