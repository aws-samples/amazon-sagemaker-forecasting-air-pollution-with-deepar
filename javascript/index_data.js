console.log('indexing');
function index_data(db_name, table, data) {
    let openRequest = indexedDB.open(db_name, 1);
    openRequest.onerror = function() {
        console.error("Error", openRequest.error);
    };

    openRequest.onsuccess = function() {
        let db = openRequest.result; 
        let transaction = db.transaction(table, "readwrite");  
        let store = transaction.objectStore(table);
        
        for (let d of data) {
            let request = store.put(d);
            request.onerror = function() {
                console.log("Error " + request.error);  
            };
        };
    };
};
