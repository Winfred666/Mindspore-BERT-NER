<template>
  <div class="relative h-screen flex flex-col items-center justify-start mx-4">
    <div class="text-lg mt-4 mb-1">聊天记录连贯性与实体识别</div>
    <div class="text-sm text-gray-700 mb-4">请在 localhost:5000 部署支持跨域的后端</div>
    <div class="flex-grow mb-8 w-full flex flex-row gap-4">
      <MyChat />
        <div class="flex flex-col gap-10">
        <Analysis />
        <DisplayTag />
        <ClearButton />
        </div>
    </div>
  </div>
</template>

<script setup>
import MyChat from './MyChat.vue';
import Analysis from './Analysis.vue';

import ClearButton from './ClearButton.vue';
import DisplayTag from './DisplayTag.vue';
</script>

<style scoped></style>
