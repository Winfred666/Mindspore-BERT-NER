import { ref, computed } from 'vue'
import { defineStore } from "pinia";


const dummyThreads = [{
    "id": 0,
    "abstract": "This is a test abstract",
    "messageIDs": [0, 1],
}, {
    "id": 1,
    "abstract": "This is another test abstract",
    "messageIDs": [2, 3],
}]

const dummyNERs = [{
    "type": "time",
    "messageID": 0,
    "start": 0,
    "end": 2,
    "text": "今天"
}, 
{
    "type": "location",
    "messageID": 2,
    "start": 3,
    "end": 7,
    "text": "体育公园"
},
{
    "type": "person",
    "messageID": 2,
    "start": 0,
    "end": 2,
    "text": "我们"
}]

export const useThreadStore = defineStore("thread", () => {
    const all_threads = ref(dummyThreads);
    // a thread need to have id, abstract(optional), messageIDs list, NERs list.
    // and one NER is a object with key 'type', 'messageID' 'start'(char index), 'end'(char index)
    
    const getThread = (threadID) => {
        return all_threads.value.find(thread => thread.id === threadID)
    }

    const addThread = (thread) => {
        all_threads.value.push(thread)
    }
    const clear = () => {
        all_threads.value = []
    }
    const addMessageToThread = (messageID, threadID) => {
        const thread = getThread(threadID)
        if(!thread) return false
        if(!thread.messageIDs) thread.messageIDs = []
        thread.messageIDs.push(messageID)
        return true
    }
    const addNERToThread = (ner, threadID) => {
        const thread = getThread(threadID)
        if(!thread) return false
        if(!thread.NERs) thread.NERs = []
        thread.NERs.push(ner)
        return true
    }
    return {all_threads, getThread, addThread, clear, addMessageToThread, addNERToThread}
})

export const useNERStore = defineStore("NER", () => {
    const all_NERs = ref(dummyNERs);
    const addNER = (ner) => {
        all_NERs.value.push(ner)
    }
    const getNERsByMessageID = (messageID) => {
        return all_NERs.value.filter(ner => ner.messageID === messageID)
    }
    const clear = () => {
        all_NERs.value = []
    }
    return {all_NERs, addNER, clear, getNERsByMessageID}
})