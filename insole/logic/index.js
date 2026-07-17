async function connectToInsole() {
    try {
        const device = await navigator.bluetooth.requestDevice({
            filters: [{services: ['insole-data']}],
            optionalServices: ['insole-data']
        })

        

    } catch (error) {
        alert(error);
    }
    
};