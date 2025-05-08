<template>
  <div class="flex flex-col justify-between w-1/2 rounded-lg shadow-lg  py-4 px-2 bg-white">
    <div class="flex flex-col w-full gap-2 overflow-y-auto px-4" :style='{ "maxHeight": "80vh" }'>
      <div v-for="dialog in dialogs" :key="dialog.id"
        v-show='activeMessageIDs.length == 0 || activeMessageIDs.includes(dialog.id)'>
        <!-- if speaker is "me", show right, else show at left. Show avator as well -->
        <div @mouseenter="() => currentDialog = dialog" @mouseleave="() => currentDialog = undefined"
          :class='"flex w-full items-center gap-1" + (dialog.name == "me" ? " flex-row-reverse justify-self-end" : " flex-row justify-self-start") 
            + ((currentDialog && dialog.related) ? 
              ( (dialog.id == currentDialog.id || currentDialog.related.some(item=>item.id == dialog.id)) ? " shadow-lg" : " opacity-25") : "")'>
          <!-- also the avator -->
          <n-avatar round :color="getColor(dialog.name)">{{ dialog.name }}</n-avatar>
          <NTooltip trigger="hover">
            <template #trigger>
              <div class="w-fit bg-slate-100 text-gray-700 p-2 rounded-lg m-1 break-words" :style='{ "maxWidth": "75%" }'
                v-html="getMessageHTML(dialog.id, dialog.content)">
              </div>
            </template>
            关联对话：{{ dialog.related?.length ?? 0 }}
          </NTooltip>

        </div>
      </div>
    </div>
    <!-- an input block with send button -->
    <div class="flex justify-center flex-row w-full gap-2 align-middle">
      <n-input size="large" class="w-4/5" v-model:value="myInput" type="text" placeholder="请输入消息"
        @keydown="enterSending"></n-input>
      <n-button type="primary" size="large" @click="sendMessage">发送</n-button>
    </div>
  </div>
</template>

<script setup>
import { storeToRefs } from 'pinia';
import { useDialogStore } from '../stores/dialog';
import { useNERStore } from '../stores/result';

import { ref } from 'vue';
import { NInput, NButton, NAvatar, useMessage, NTooltip } from 'naive-ui';
import { getColor, emitter } from './utils';

const dialogStore = useDialogStore();
const { dialogs } = storeToRefs(dialogStore);
const nerStore = useNERStore();
// const {all_NERs} = storeToRefs(nerStore); // use this to highlight NERs in messages.

const currentDialog = ref(undefined);

emitter.on("thread-clicked", (thread) => {
  // get messageIDs out to highlight it.
  const { messageIDs } = thread;
  activeMessageIDs.value = messageIDs;
});

emitter.on("thread-closed", () => {
  activeMessageIDs.value = [];
});

const activeMessageIDs = ref([]);

const myInput = ref('');

const enterSending = (e) => {
  if (e.key === 'Enter') {
    sendMessage();
  }
};

const UIMessage = useMessage();

const sendMessage = () => {
  if (myInput.value.length < 2) {
    UIMessage.warning('发送消息太短', { duration: 2000 });
    return;
  }
  const new_message = {
    id: dialogs.value.length,
    name: 'me',
    content: myInput.value,
  }
  dialogStore.addMessage(new_message);
  myInput.value = ''; // clear the input
};



// important, triky way to render the message content with NERs highlighted.
const getMessageHTML = (id, content) => {
  // this already filtered out NERs for this message.
  let NERs = nerStore.getNERsByMessageID(id);
  let result = content;
  // replace all the < and > in the content to avoid XSS attack.
  content = content.replace(/</g, "&lt;").replace(/>/g, "&gt;");
  // sort from the end to the start, so that the index won't be changed.
  NERs = NERs.sort((a, b) => {
    return b.start - a.start;
  });
  NERs.forEach(ner => {
    const start = ner.start;
    const end = ner.end; //remember, end is exclusive !!!
    const type = ner.type ?? "other";
    const span = `<span class="entity entity-${type.toLowerCase()}">${content.slice(start, end)}</span>`;
    result = result.slice(0, start) + span + result.slice(end);
  });
  return result;
};

</script>
