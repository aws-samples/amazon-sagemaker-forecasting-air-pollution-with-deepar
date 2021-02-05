let openRequest = indexedDB.open('openaq', 1);

openRequest.onupgradeneeded = function() {
    let db = openRequest.result;
    if (!db.objectStoreNames.contains('actuals')) {
            db.createObjectStore('actuals', {keyPath: 'id'});
            db.createObjectStore('predictions', {keyPath: 'id'});
    }
};