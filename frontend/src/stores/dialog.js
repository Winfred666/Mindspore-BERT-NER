import { ref, computed } from 'vue'
import { defineStore } from "pinia";



const dummyDialog = [{
    "id": 0,
    "name": "小明",
    "content": "今天天气不错！"
}, {
    "id": 1,
    "name": "小红",
    "content": "的确，一天都是好天气。"
}, {
    "id": 2,
    "name": "小明",
    "content": "我们去体育公园打球吧！"
}, {
    "id": 3,
    "name": "小红",
    "content": "好啊！"
}]

export const useDialogStore = defineStore("dialog", () => {
    const dialogs = ref(dummyDialog)
    const addMessage = (message) => {
        dialogs.value.push(message)
    }
    const setNewDialogs = (newDialogs) => {
        dialogs.value = newDialogs
    }
    const clear = () => {
        // dialogs.value.splice(index, 1) //remove dialog at specifit index
        dialogs.value = [] //clear all dialogs
    }
    return { dialogs, addMessage, clear, setNewDialogs }
})