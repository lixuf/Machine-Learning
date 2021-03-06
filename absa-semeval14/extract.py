def extract_items(text_in,text_in_un):
    _k1, _k2 = subject_model.predict([text_in_un, text_in])
    _k1, _k2 = _k1[0, :, 0], _k2[0, :, 0]
    _k1, _k2 = np.where(_k1 > 0.5)[0], np.where(_k2 > 0.4)[0]
    _subjects = []
    for i in _k1:
        j = _k2[_k2 >= i]
        if len(j) > 0:
            j = j[0]
            _subject = text_in_un[i: j+1]
            _subjects.append(_subject)
    return tuple(_subjects)