const createAddUniqueNumber = (generateUniqueNumber) => {
    return (set) => {
        const number = generateUniqueNumber(set);
        set.add(number);
        return number;
    };
};

const createCache = (lastNumberWeakMap) => {
    return (collection, nextNumber) => {
        lastNumberWeakMap.set(collection, nextNumber);
        return nextNumber;
    };
};

/*
 * The value of the constant Number.MAX_SAFE_INTEGER equals (2 ** 53 - 1) but it
 * is fairly new.
 */
const MAX_SAFE_INTEGER = Number.MAX_SAFE_INTEGER === undefined ? 9007199254740991 : Number.MAX_SAFE_INTEGER;
const TWO_TO_THE_POWER_OF_TWENTY_NINE = 536870912;
const TWO_TO_THE_POWER_OF_THIRTY = TWO_TO_THE_POWER_OF_TWENTY_NINE * 2;
const createGenerateUniqueNumber = (cache, lastNumberWeakMap) => {
    return (collection) => {
        const lastNumber = lastNumberWeakMap.get(collection);
        /*
         * Let's try the cheapest algorithm first. It might fail to produce a new
         * number, but it is so cheap that it is okay to take the risk. Just
         * increase the last number by one or reset it to 0 if we reached the upper
         * bound of SMIs (which stands for small integers). When the last number is
         * unknown it is assumed that the collection contains zero based consecutive
         * numbers.
         */
        let nextNumber = lastNumber === undefined ? collection.size : lastNumber < TWO_TO_THE_POWER_OF_THIRTY ? lastNumber + 1 : 0;
        if (!collection.has(nextNumber)) {
            return cache(collection, nextNumber);
        }
        /*
         * If there are less than half of 2 ** 30 numbers stored in the collection,
         * the chance to generate a new random number in the range from 0 to 2 ** 30
         * is at least 50%. It's benifitial to use only SMIs because they perform
         * much better in any environment based on V8.
         */
        if (collection.size < TWO_TO_THE_POWER_OF_TWENTY_NINE) {
            while (collection.has(nextNumber)) {
                nextNumber = Math.floor(Math.random() * TWO_TO_THE_POWER_OF_THIRTY);
            }
            return cache(collection, nextNumber);
        }
        // Quickly check if there is a theoretical chance to generate a new number.
        if (collection.size > MAX_SAFE_INTEGER) {
            throw new Error('Congratulations, you created a collection of unique numbers which uses all available integers!');
        }
        // Otherwise use the full scale of safely usable integers.
        while (collection.has(nextNumber)) {
            nextNumber = Math.floor(Math.random() * MAX_SAFE_INTEGER);
        }
        return cache(collection, nextNumber);
    };
};

const LAST_NUMBER_WEAK_MAP = new WeakMap();
const cache = createCache(LAST_NUMBER_WEAK_MAP);
const generateUniqueNumber = createGenerateUniqueNumber(cache, LAST_NUMBER_WEAK_MAP);
const addUniqueNumber = createAddUniqueNumber(generateUniqueNumber);

const isMessagePort = (sender) => {
    return typeof sender.start === 'function';
};

const PORT_MAP = new WeakMap();

const extendBrokerImplementation = (partialBrokerImplementation) => ({
    ...partialBrokerImplementation,
    connect: ({ call }) => {
        return async () => {
            const { port1, port2 } = new MessageChannel();
            const portId = await call('connect', { port: port1 }, [port1]);
            PORT_MAP.set(port2, portId);
            return port2;
        };
    },
    disconnect: ({ call }) => {
        return async (port) => {
            const portId = PORT_MAP.get(port);
            if (portId === undefined) {
                throw new Error('The given port is not connected.');
            }
            await call('disconnect', { portId });
        };
    },
    isSupported: ({ call }) => {
        return () => call('isSupported');
    }
});

const ONGOING_REQUESTS = new WeakMap();
const createOrGetOngoingRequests = (sender) => {
    if (ONGOING_REQUESTS.has(sender)) {
        // @todo TypeScript needs to be convinced that has() works as expected.
        return ONGOING_REQUESTS.get(sender);
    }
    const ongoingRequests = new Map();
    ONGOING_REQUESTS.set(sender, ongoingRequests);
    return ongoingRequests;
};
const createBroker = (brokerImplementation) => {
    const fullBrokerImplementation = extendBrokerImplementation(brokerImplementation);
    return (sender) => {
        const ongoingRequests = createOrGetOngoingRequests(sender);
        sender.addEventListener('message', (({ data: message }) => {
            const { id } = message;
            if (id !== null && ongoingRequests.has(id)) {
                const { reject, resolve } = ongoingRequests.get(id);
                ongoingRequests.delete(id);
                if (message.error === undefined) {
                    resolve(message.result);
                }
                else {
                    reject(new Error(message.error.message));
                }
            }
        }));
        if (isMessagePort(sender)) {
            sender.start();
        }
        const call = (method, params = null, transferables = []) => {
            return new Promise((resolve, reject) => {
                const id = generateUniqueNumber(ongoingRequests);
                ongoingRequests.set(id, { reject, resolve });
                if (params === null) {
                    sender.postMessage({ id, method }, transferables);
                }
                else {
                    sender.postMessage({ id, method, params }, transferables);
                }
            });
        };
        const notify = (method, params, transferables = []) => {
            sender.postMessage({ id: null, method, params }, transferables);
        };
        let functions = {};
        for (const [key, handler] of Object.entries(fullBrokerImplementation)) {
            functions = { ...functions, [key]: handler({ call, notify }) };
        }
        return { ...functions };
    };
};

export { addUniqueNumber as a, createBroker as c, generateUniqueNumber as g };
//# sourceMappingURL=module-a4efca6e.js.map
